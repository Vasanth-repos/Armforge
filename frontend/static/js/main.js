let isRunning = false;

async function runBenchmark() {
  if (isRunning) return;
  const mode = document.getElementById('modeSelect').value;
  const model_name = document.getElementById('modelSelect').value;
  const runBtn = document.getElementById('runBtn');
  
  isRunning = true;
  runBtn.disabled = true;
  runBtn.innerHTML = '⏳ Running Benchmark...';
  document.getElementById('jobStatusBadge').innerText = 'RUNNING';

  try {
    const res = await fetch('/api/benchmark/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, model_name })
    });
    const data = await res.json();
    pollStatus();
  } catch (e) {
    alert('Error triggering benchmark: ' + e);
    isRunning = false;
    runBtn.disabled = false;
    runBtn.innerHTML = '▶ Launch Benchmark Suite';
  }
}

async function pollStatus() {
  const interval = setInterval(async () => {
    try {
      const res = await fetch('/api/benchmark/status');
      const data = await res.json();
      
      const term = document.getElementById('terminalLog');
      term.innerHTML = data.logs.map(l => `<div class="terminal-line">${l}</div>`).join('');
      term.scrollTop = term.scrollHeight;

      document.getElementById('jobStatusBadge').innerText = data.status.toUpperCase();

      if (data.status === 'completed' || data.status === 'error') {
        clearInterval(interval);
        isRunning = false;
        const runBtn = document.getElementById('runBtn');
        runBtn.disabled = false;
        runBtn.innerHTML = '▶ Launch Benchmark Suite';
        fetchMetrics();
      }
    } catch (e) {
      console.error(e);
    }
  }, 1000);
}

async function fetchMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    document.getElementById('cpuVal').innerText = data.cpu_pct + '%';
    document.getElementById('cpuBar').style.width = data.cpu_pct + '%';
    document.getElementById('memVal').innerText = `${data.mem_used_gb} / ${data.mem_total_gb} GB`;
  } catch (e) {}
}

function exportReport() {
  window.location.href = '/api/benchmark/export?format=markdown';
}

setInterval(fetchMetrics, 4000);
