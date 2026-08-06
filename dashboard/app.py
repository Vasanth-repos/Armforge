"""
FastAPI dashboard — live metrics, interactive prompt playground, & benchmark results.
Run: uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
"""
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import psutil, json, glob, os, time, requests
from datetime import datetime

app = FastAPI(title="ArmForge Dashboard & Playground")
templates = Jinja2Templates(directory="dashboard/templates")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    port: int = 8000

def get_arm_features():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('Features'):
                    return line.split(':')[1].strip().split()
    except Exception:
        pass
    return []

def load_results():
    out = []
    for fp in sorted(glob.glob("results/bench_*.json")):
        try:
            with open(fp) as f: out.append(json.load(f))
        except Exception:
            pass
    return out

def get_optimal_threads():
    try:
        with open('results/optimal_threads.txt') as f: return f.read().strip()
    except Exception:
        return "N/A"

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Clean handler for browser favicon requests."""
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    features = get_arm_features()
    active   = [f for f in ["i8mm","asimddp","sve","asimd"] if f in features]
    mem      = psutil.virtual_memory()
    return templates.TemplateResponse("index.html", {
        "request":          request,
        "platform":         "arm64",
        "cores":            os.cpu_count(),
        "optimal_threads":  get_optimal_threads(),
        "cpu_pct":          round(psutil.cpu_percent(interval=0.5), 1),
        "mem_used_gb":      round(mem.used / 1e9, 1),
        "mem_total_gb":     round(mem.total / 1e9, 1),
        "arm_features":     active,
        "results":          load_results()[-8:],
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.get("/api/metrics")
async def metrics():
    mem = psutil.virtual_memory()
    return {
        "platform":         "arm64",
        "arch":             os.uname().machine,
        "cpu_count":        os.cpu_count(),
        "optimal_threads":  get_optimal_threads(),
        "cpu_pct":          round(psutil.cpu_percent(interval=0.5), 1),
        "mem_used_gb":      round(mem.used / 1e9, 2),
        "mem_total_gb":     round(mem.total / 1e9, 2),
        "arm_features":     get_arm_features(),
        "bench_results":    load_results(),
        "timestamp":        datetime.now().isoformat(),
    }

@app.post("/api/generate")
async def generate_stream(req: GenerateRequest):
    """Proxy streaming completion request to local llama-server."""
    target_url = f"http://localhost:{req.port}/v1/completions"

    def stream_generator():
        payload = {
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "stream": True
        }
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0

        try:
            r = requests.post(target_url, json=payload, stream=True, timeout=120)
            for chunk in r.iter_lines():
                if chunk:
                    line = chunk.decode('utf-8')
                    if line.startswith("data: "):
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start_time
                        token_count += 1
                        yield f"{line}\n\n"
            
            total_elapsed = time.perf_counter() - start_time
            tps = token_count / total_elapsed if total_elapsed > 0 else 0
            stats = {
                "ttft_ms": round((first_token_time or 0) * 1000, 1),
                "tokens": token_count,
                "elapsed_s": round(total_elapsed, 2),
                "tps": round(tps, 2)
            }
            yield f"data: [STATS] {json.dumps(stats)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
