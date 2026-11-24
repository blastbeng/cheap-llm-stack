#!/usr/bin/env bash
#
# update-llama.cpp.sh - rebuild llama.cpp when upstream publishes a new tag.
#
# Run weekly by llama-swap-update.timer (Sun 03:00) or by hand:
#   sudo ./update-llama.cpp.sh              # newest upstream tag
#   sudo LLAMA_CPP_TAG=b8123 ./update-llama.cpp.sh
#
# The build itself is delegated to install.sh so the compile flags can never
# drift between a first install and an update (-march=armv8.2-a+dotprod+fp16,
# CPU-only: no CUDA/Vulkan/NPU backends on this SoC). A rebuild is atomic from
# the service's point of view: the old binary keeps serving until the new one
# is in place and llama-swap is restarted.
#
set -euo pipefail

STACK_DIR="${STACK_DIR:-/opt/cheap-llm-stack}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${STACK_DIR}/.llama-cpp-version"
REPO="ggml-org/llama.cpp"

log()  { printf '\033[1;32m[update]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo $0)"
[ -x "${STACK_DIR}/bin/llama-server" ] || die "${STACK_DIR}/bin/llama-server missing; run install.sh first"

current="$(cat "${VERSION_FILE}" 2>/dev/null || echo none)"

if [ -n "${LLAMA_CPP_TAG:-}" ]; then
    target="${LLAMA_CPP_TAG}"
else
    # /releases/latest is one request and gives the newest non-prerelease tag;
    # the tags endpoint is rate-limited harder and can return RC candidates.
    target="$(curl -fsSL --max-time 30 "https://api.github.com/repos/${REPO}/releases/latest" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')" \
        || die "cannot query GitHub releases for ${REPO}"
fi

log "installed: ${current}    upstream: ${target}"
if [ "${target}" = "${current}" ]; then
    log "already up to date"
    exit 0
fi

was_active=0
systemctl is-active --quiet llama-swap.service && was_active=1

log "rebuilding llama.cpp ${target} (this takes ~20-40 min on this board)"
if LLAMA_CPP_TAG="${target}" "${SRC_DIR}/install.sh" --skip-models --no-apt; then
    log "build ok: $(cat "${VERSION_FILE}")"
else
    warn "build failed - keeping the running binary (${current})"
    exit 1
fi

if [ "${was_active}" = 1 ]; then
    log "restarting llama-swap"
    systemctl restart llama-swap.service
    for _ in $(seq 1 60); do
        if curl -fsS --max-time 3 "http://127.0.0.1:10001/v1/models" >/dev/null 2>&1; then
            log "llama-swap healthy on ${target}"
            exit 0
        fi
        sleep 2
    done
    die "llama-swap did not come back after the update; check journalctl -u llama-swap"
fi

log "done (service was not running, so it was not restarted)"
