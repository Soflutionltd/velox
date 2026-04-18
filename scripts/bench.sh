#!/usr/bin/env bash
# Velox throughput benchmark.
#
# Usage:
#   ./scripts/bench.sh                          # default: Qwen3-0.6B, 200 tok, 5 runs
#   ./scripts/bench.sh <model> <max_tokens> <runs>
#
# Requires a running velox server (default http://localhost:8000).

set -euo pipefail

MODEL="${1:-Qwen3-0.6B}"
MAX_TOKENS="${2:-200}"
RUNS="${3:-5}"
HOST="${VELOX_HOST:-http://localhost:8000}"

PROMPT='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Write a 200-word essay about the importance of pure-Rust LLM servers."}],"max_tokens":'"$MAX_TOKENS"',"temperature":0}'

echo "==========================================="
echo " Velox bench"
echo " host  = $HOST"
echo " model = $MODEL"
echo " tokens/run = $MAX_TOKENS"
echo " runs  = $RUNS"
echo "==========================================="

python3 - <<'PYEOF'
import json, time, urllib.request, sys, os
host = os.environ.get("VELOX_HOST", "http://localhost:8000")
prompt = os.environ["PROMPT"]
runs = int(os.environ["RUNS"])
total_toks = 0
total_time = 0.0
print()
print(f"{'run':>4}  {'time(s)':>8}  {'tokens':>7}  {'tok/s':>7}")
print("-" * 32)
for i in range(1, runs + 1):
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=prompt.encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
    t1 = time.time()
    data = json.loads(body, strict=False)
    toks = data["usage"]["completion_tokens"]
    elapsed = t1 - t0
    tps = toks / elapsed if elapsed > 0 else 0.0
    total_toks += toks
    total_time += elapsed
    print(f"{i:>4}  {elapsed:>8.2f}  {toks:>7}  {tps:>7.1f}")
print("-" * 32)
avg = total_toks / total_time if total_time > 0 else 0.0
print(f"avg                        {avg:>7.1f}")
PYEOF
