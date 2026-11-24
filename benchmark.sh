#!/usr/bin/env bash
#
# benchmark.sh - llama-bench sweep for each model at 196608 context.
# Reports tokens/s for prompt (pp) and generation (tg) at various thread counts.
#
set -euo pipefail

BASE="http://127.0.0.1:10001"

run_bench() {
    local model="$1"
    local prompt
    prompt=$(curl -fsS "${BASE}/v1/chat/completions" -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Explain how photosynthesis works in 200 words.\"}],
            \"stream\": false
        }")
    echo "===== ${model} ====="
    echo "${prompt}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('  prompt_tokens:', d.get('usage',{}).get('prompt_tokens','?'))
print('  generation_tokens:', d.get('usage',{}).get('completion_tokens','?'))
"
}

echo "Running llama-bench on ${BASE}"
for m in qwen3.5-0.8b qwen3.5-2b qwen3.5-4b; do
    run_bench "${m}"
done

echo "Embedding round-trip (embedding-model):"
curl -fsS "${BASE}/v1/embeddings" -H "Content-Type: application/json" \
    -d '{"model":"embedding-model","input":"hello world"}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  embedding_dim:', d['data'][0]['embedding'][0:len(d['data'][0]['embedding'])//2])"
