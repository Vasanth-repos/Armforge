"""
Interactive CLI LLM Chat Client with Live Streaming & Performance Timer.

Usage:
  python inference/demo_chat.py --port 8000
"""
import sys, os, time, json, argparse, requests

def chat_loop(port):
    base_url = f"http://localhost:{port}/v1/completions"
    print("============================================================")
    print(" 🦾 ArmForge Interactive CLI LLM Chat Client")
    print(f" Target Server: {base_url}")
    print(" Type 'exit' or 'quit' to end the session.")
    print("============================================================\n")

    while True:
        try:
            prompt = input("\nUser > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Exiting chat session.")
                break

            print("\nArmForge > ", end="", flush=True)

            payload = {
                "prompt": prompt,
                "max_tokens": 256,
                "stream": True
            }

            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0

            with requests.post(base_url, json=payload, stream=True, timeout=120) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    text_line = line.decode('utf-8')
                    if text_line.startswith("data: "):
                        if text_line == "data: [DONE]":
                            break
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start_time
                        token_count += 1
                        try:
                            data = json.loads(text_line[6:])
                            token_text = data.get("choices", [{}])[0].get("text", "")
                            print(token_text, end="", flush=True)
                        except Exception:
                            pass

            total_elapsed = time.perf_counter() - start_time
            tps = token_count / total_elapsed if total_elapsed > 0 else 0
            ttft_ms = (first_token_time or 0) * 1000

            print(f"\n\n  [Metrics] TTFT: {ttft_ms:.0f} ms | Throughput: {tps:.2f} tok/s | Tokens: {token_count} | Time: {total_elapsed:.2f}s")

        except KeyboardInterrupt:
            print("\nChat interrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000, help="Port of active llama-server")
    args = ap.parse_args()
    chat_loop(args.port)
