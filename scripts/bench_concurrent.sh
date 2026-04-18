#!/usr/bin/env bash
# Concurrent throughput benchmark for Velox.
#
# Spawns N parallel chat-completion requests to /v1/chat/completions and
# reports total tokens, wall-clock, and aggregate tok/s. Compares with a
# single-request run for context.
#
# Usage:
#   ./scripts/bench_concurrent.sh [N=8] [MAX_TOKENS=64] [URL=http://localhost:8000]

set -euo pipefail

N="${1:-8}"
MAX_TOKENS="${2:-64}"
URL="${3:-http://localhost:8000}"
MODEL="${MODEL:-Qwen3-0.6B}"

PROMPTS=(
  "Write a haiku about the ocean."
  "Explain Newton's first law in one short sentence."
  "List three primary colors."
  "Translate 'good morning' to Japanese."
  "What is 17 times 4?"
  "Describe a cat in one sentence."
  "Name three planets in our solar system."
  "Suggest a name for a coffee shop."
  "What's the capital of Australia?"
  "Give me a short tip for better sleep."
  "Recommend a book about Rust."
  "How do you boil an egg?"
  "Define 'serendipity'."
  "Compose a one-line poem about coding."
  "What is HTTP?"
  "Quick exercise idea for the office."
)

build_payload() {
  local prompt="$1"
  python3 -c "
import json
print(json.dumps({
  'model': '$MODEL',
  'messages': [{'role': 'user', 'content': $prompt!r}],
  'max_tokens': $MAX_TOKENS,
  'temperature': 0
}))
" <<EOF
$prompt
EOF
}

run_one() {
  local idx="$1"
  local prompt="${PROMPTS[$((idx % ${#PROMPTS[@]}))]}"
  local payload
  payload=$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAX_TOKENS,'temperature':0}))" "$prompt")
  curl -s -o "/tmp/velox-bench-$idx.json" -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload"
}

extract_tokens() {
  python3 -c "
import json, sys, glob
total = 0
for f in glob.glob('/tmp/velox-bench-*.json'):
    try:
        d = json.loads(open(f).read())
        total += d.get('usage', {}).get('completion_tokens', 0)
    except Exception:
        pass
print(total)
"
}

echo "================================================================="
echo " Velox concurrent benchmark: $N parallel requests, $MAX_TOKENS max_tokens each"
echo "================================================================="

# Warm up.
echo "Warming up..."
run_one 0 > /dev/null
rm -f /tmp/velox-bench-*.json

# === Single-request baseline ===
echo
echo "[1/2] Single-request baseline..."
START=$(python3 -c 'import time; print(time.time())')
run_one 0
END=$(python3 -c 'import time; print(time.time())')
SINGLE_TOKENS=$(extract_tokens)
SINGLE_DUR=$(python3 -c "print(f'{$END - $START:.3f}')")
SINGLE_TPS=$(python3 -c "print(f'{$SINGLE_TOKENS / ($END - $START):.1f}')")
echo "  ${SINGLE_TOKENS} tokens in ${SINGLE_DUR}s = ${SINGLE_TPS} tok/s"
rm -f /tmp/velox-bench-*.json

# === N concurrent ===
echo
echo "[2/2] $N concurrent requests..."
START=$(python3 -c 'import time; print(time.time())')
PIDS=()
for i in $(seq 0 $((N-1))); do
  run_one "$i" &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done
END=$(python3 -c 'import time; print(time.time())')
N_TOKENS=$(extract_tokens)
N_DUR=$(python3 -c "print(f'{$END - $START:.3f}')")
N_TPS=$(python3 -c "print(f'{$N_TOKENS / ($END - $START):.1f}')")
echo "  ${N_TOKENS} tokens in ${N_DUR}s = ${N_TPS} tok/s aggregate"
SCALE=$(python3 -c "print(f'{$N_TPS / $SINGLE_TPS:.2f}')")
echo
echo "================================================================="
echo " Throughput scaling: ${SCALE}x at concurrency=$N"
echo "================================================================="
rm -f /tmp/velox-bench-*.json
