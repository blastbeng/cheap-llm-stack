#!/usr/bin/env bash
#
# healthcheck.sh - quick status check for the whole stack.
#
set -euo pipefail

BASE="${LLAMA_SWAP_URL:-http://127.0.0.1:10001}"
fail=0

check() {
    local name="$1" url="$2"
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
        printf '  [ok]   %s\n' "$name"
    else
        printf '  [FAIL] %s (%s)\n' "$name" "$url"
        fail=1
    fi
}

echo "llama-swap health check (${BASE})"
check "orchestrator /v1/models" "${BASE}/v1/models"

echo "Model availability:"
curl -fsS "${BASE}/v1/models" 2>/dev/null | python3 -c "
import sys,json
try:
    for m in json.load(sys.stdin)['data']:
        print('  -', m['id'])
except Exception:
    print('  (cannot list models)')
"

echo "System:"
systemctl is-active llama-swap.service && echo "  llama-swap.service active" || { echo "  llama-swap.service DOWN"; fail=1; }
systemctl is-active llama-swap-update.timer && echo "  update timer active" || echo "  update timer inactive"

echo "Memory:"
free -h | sed 's/^/  /'

exit "${fail}"
