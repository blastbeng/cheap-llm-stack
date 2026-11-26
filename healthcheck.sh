#!/usr/bin/env bash
#
# healthcheck.sh - one glance at the whole stack.
#   router (:10000) -> llama-swap (:10001) -> models
#
#   ./healthcheck.sh            # endpoints, models, embeddings, vision, services
#   ./healthcheck.sh --tools    # also the 8-case tool-calling battery (~3 min)
#
set -uo pipefail

ROUTER="${ROUTER_URL:-http://127.0.0.1:10000}"
SWAP="${LLAMA_SWAP_URL:-http://127.0.0.1:10001}"
fail=0

check() {
    local name="$1" url="$2"
    if curl -fsS --max-time 8 "$url" >/dev/null 2>&1; then
        printf '  [ok]   %s\n' "${name}"
    else
        printf '  [FAIL] %s (%s)\n' "${name}" "${url}"
        fail=1
    fi
}

echo "router ${ROUTER}"
if curl -fsS --max-time 8 "${ROUTER}/health" 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("  status=%s prototypes_ready=%s" % (d.get("status"), d.get("prototypes_ready")))
if d.get("upstream_error"):
    print("  upstream_error:", d["upstream_error"]); sys.exit(1)
' 2>/dev/null; then :; else
    echo "  [FAIL] router /health"; fail=1
fi

echo "llama-swap ${SWAP}"
check "orchestrator /v1/models" "${SWAP}/v1/models"

# Two lists on purpose. llama-swap knows the real model ids; `auto` exists only in
# the router's answer, so requiring it against ${SWAP} would fail forever.
echo "models (llama-swap)"
curl -fsS --max-time 8 "${SWAP}/v1/models" 2>/dev/null | python3 -c '
import json, sys
try:
    ids = [m["id"] for m in json.load(sys.stdin)["data"]]
    for i in ids:
        print("  -", i)
    missing = [m for m in ("embedding-model", "router", "reasoner", "vision")
               if m not in ids]
    if missing:
        print("  [FAIL] not configured:", ", ".join(missing)); sys.exit(1)
except Exception as exc:
    print("  [FAIL] cannot list models:", exc); sys.exit(1)
' || fail=1

echo "models (router, what a client sees)"
curl -fsS --max-time 8 "${ROUTER}/v1/models" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)["data"]
    ids = [m["id"] for m in d]
    print("  -", ", ".join(sorted(ids)))
    missing = [m for m in ("auto", "embedding-model", "router", "reasoner", "vision")
               if m not in ids]
    if missing:
        print("  [FAIL] router is not advertising:", ", ".join(missing)); sys.exit(1)
    auto = [m for m in d if m["id"] == "auto"]
    cap = (auto[0].get("capabilities") or {}) if auto else {}
    if not cap.get("function_calling") or not cap.get("vision"):
        print("  [FAIL] auto does not advertise function_calling + vision"); sys.exit(1)
except Exception as exc:
    print("  [FAIL] cannot list models through the router:", exc); sys.exit(1)
' || fail=1

echo "embedding-model (mandatory, always online)"
if curl -fsS --max-time 60 "${ROUTER}/v1/embeddings" -H 'Content-Type: application/json' \
     -d '{"model":"embedding-model","input":"ping"}' 2>/dev/null \
   | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  dims=%d" % len(d["data"][0]["embedding"]))' 2>/dev/null; then :; else
    echo "  [FAIL] embeddings"; fail=1
fi

echo "vision"
if curl -fsS --max-time 300 "${ROUTER}/v1/chat/completions" -H 'Content-Type: application/json' \
     -d '{"model":"vision","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"ok?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="}}]}]}' \
     >/dev/null 2>&1; then
    echo "  [ok]   vision accepts image_url"
else
    echo "  [FAIL] vision request"; fail=1
fi

if [ "${1:-}" = "--tools" ]; then
    echo "tool calling (8 cases, via the router)"
    if [ -x "${0%/*}/scripts/toolcheck.py" ]; then
        "${0%/*}/scripts/toolcheck.py" --url "${ROUTER}" || fail=1
    else
        python3 "$(dirname "$0")/scripts/toolcheck.py" --url "${ROUTER}" || fail=1
    fi
fi

echo "services"
for u in llama-swap.service llm-router.service; do
    if systemctl is-active --quiet "${u}"; then echo "  [ok]   ${u}"; else echo "  [FAIL] ${u}"; fail=1; fi
done
if systemctl is-active --quiet llama-swap-update.timer; then
    echo "  [ok]   llama-swap-update.timer (next: $(systemctl show -p NextElapseUSecRealtime --value llama-swap-update.timer 2>/dev/null))"
else
    echo "  [warn] llama-swap-update.timer inactive"
fi

echo "system"
free -h | sed 's/^/  /'
awk '{printf "  temp: %.0f C\n", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null

exit "${fail}"
