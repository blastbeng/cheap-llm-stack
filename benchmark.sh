#!/usr/bin/env bash
#
# benchmark.sh - measure this board, not a datasheet.
#
#   ./benchmark.sh            # llama-bench: pp512 / tg128 for every resident model
#   ./benchmark.sh --http     # end-to-end latency through the router (real tok/s)
#   ./benchmark.sh --embed    # embedding model latency + dimensions
#   ./benchmark.sh --all
#
# The pp/tg numbers from llama-bench are the ones that matter for tuning:
# generation (tg) is bandwidth-bound and scales with the A76 cores (2 max),
# prompt processing (pp) is compute-bound and uses all eight cores.
# llama-bench has no -tb switch: it is given a -t list and the best pp thread
# count is what llama-server uses for --threads-batch.
#
set -euo pipefail

STACK_DIR="${STACK_DIR:-/opt/cheap-llm-stack}"
BIN="${STACK_DIR}/bin/llama-bench"
MODELS="${STACK_DIR}/models"
SWAP="http://127.0.0.1:10001"
ROUTER="http://127.0.0.1:10000"

THREADS="${THREADS:-2}"
BATCH_THREADS="${BATCH_THREADS:-8}"

DO_BENCH=0; DO_HTTP=0; DO_EMBED=0
[ $# -eq 0 ] && DO_BENCH=1
for arg in "$@"; do
    case "${arg}" in
        --bench) DO_BENCH=1 ;;
        --http)  DO_HTTP=1 ;;
        --embed) DO_EMBED=1 ;;
        --all)   DO_BENCH=1; DO_HTTP=1; DO_EMBED=1 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

log() { printf '\033[1;32m[bench]\033[0m %s\n' "$*"; }

if [ "${DO_BENCH}" = 1 ]; then
    [ -x "${BIN}" ] || { echo "missing ${BIN} - run install.sh first" >&2; exit 1; }
    log "llama-bench (threads=${THREADS}, batch=${BATCH_THREADS})"
    for gguf in \
        "Qwen3-Embedding-0.6B-Q8_0.gguf" \
        "LFM3-1.7B-QAD-Q4_0.gguf" \
        "Qwen3-0.6B-Q8_0.gguf" \
        "LFM2.5-8B-A1B-Q4_0.gguf" \
        "LFM3-VL-2B-QAD-Q4_0.gguf"
    do
        [ -s "${MODELS}/${gguf}" ] || { log "skip ${gguf} (not downloaded)"; continue; }
        log "  ${gguf}"
        # Same KV config the stack runs (q8_0 needs flash-attn), otherwise the
        # numbers describe a setup nobody deploys: f16 KV is ~2x the memory and
        # measurably slower to fill.
        "${BIN}" -m "${MODELS}/${gguf}" \
            -fa on -ctk q8_0 -ctv q8_0 \
            -t "${THREADS},${BATCH_THREADS}" \
            -p 512 -n 128 -r 2 -o md 2>/dev/null \
            | grep -E '^\|' || true
    done
fi

if [ "${DO_EMBED}" = 1 ]; then
    log "embedding model through llama-swap"
    python3 - "$SWAP" <<'PY'
import json, sys, time, urllib.request
base = sys.argv[1]
texts = ["hello world"] * 1 + ["a slightly longer sentence to embed for timing"] * 7
body = json.dumps({"model": "embedding-model", "input": texts}).encode()
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
    base + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"}), timeout=120)
d = json.loads(r.read())
ms = (time.time() - t0) * 1000
dim = len(d["data"][0]["embedding"])
print("  dims=%d  batch=%d  total=%.0f ms  per-item=%.1f ms" % (dim, len(texts), ms, ms / len(texts)))
hits = [i for i, t in enumerate(texts) if t.startswith("a slightly")]
print("  batch grouping looks %s" % ("ok" if hits else "odd"))
PY
fi

if [ "${DO_HTTP}" = 1 ]; then
    log "end-to-end through the router (router auto-routes, headers show why)"
    for probe in "what is the capital of france|auto" \
                 "prove step by step why the square root of 2 is irrational|auto" \
                 "def add(a, b): return a + b  -- find the bug|auto"
    do
        prompt="${probe%|*}"
        python3 - "$ROUTER" "${prompt}" <<'PY'
import json, sys, time, urllib.request
base, prompt = sys.argv[1], sys.argv[2]
body = json.dumps({"model": "auto", "max_tokens": 64, "stream": False,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
    base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900)
d = json.loads(r.read())
el = time.time() - t0
u = d.get("usage", {})
ct = u.get("completion_tokens") or 0
print("  %-6s %-22s %5.1fs  %5.1f tok/s  %r" % (
    r.headers.get("X-Route"), (r.headers.get("X-Route-By") or "")[:22], el,
    (ct / el) if el > 0 else 0, prompt[:34]))
PY
    done
fi
