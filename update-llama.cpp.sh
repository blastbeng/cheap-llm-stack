#!/usr/bin/env bash
#
# update-llama.cpp.sh - rebuild llama.cpp when upstream publishes a new tag,
# then prove that the new build can still call tools.
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
# Why the build is verified and not just started
# ---------------------------------------------
# A new build fails two ways. Loudly: a removed flag makes llama-server exit
# ("flag provided but not defined: -host"), which systemd already reports. And
# silently: tool calling depends on llama.cpp recognising this model's chat
# template in common/chat.cpp (common_chat_params_init_lfm2_5 -> PEG grammar in
# chat-peg-parser.cpp). When that path changes, llama.cpp does NOT error - it
# falls back to the generic parser and the model writes the call as text inside
# `content`. /health stays green, chat still answers, agents stop working. The
# predecessor of the current model scored 1/7 for exactly that reason, without a
# single error message. So after the restart the tool battery runs; if it fails
# twice, the previous bin/ (copied to .build-backup before the build) is restored
# and the unit is left failed so `systemctl --failed` shows it.
#
set -euo pipefail

STACK_DIR="${STACK_DIR:-/opt/cheap-llm-stack}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${STACK_DIR}/.llama-cpp-version"
BACKUP_DIR="${STACK_DIR}/.build-backup"
TOOLCHECK="${SRC_DIR}/scripts/toolcheck.py"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:10000}"
# `vec` is deliberately absent: it exercises an invented identifier inside the
# arguments, which every model on this board mangles, so it warns rather than
# fails and adds 20 s without adding signal about the build.
VERIFY_CASES="${VERIFY_CASES:-tg,web,mem,wx,multi,none}"
REPO="ggml-org/llama.cpp"

log()  { printf '\033[1;32m[update]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo $0)"
[ -x "${STACK_DIR}/bin/llama-server" ] || die "${STACK_DIR}/bin/llama-server missing; run install.sh first"

current="$(cat "${VERSION_FILE}" 2>/dev/null || echo none)"

wait_swap() {
    for _ in $(seq 1 "${1:-60}"); do
        curl -fsS --max-time 3 "http://127.0.0.1:10001/v1/models" >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}

verify_tools() {
    python3 "${TOOLCHECK}" --url "${ROUTER_URL}" --only "${VERIFY_CASES}"
}

backup_build() {
    # bin/ is ~62 MB: cheaper to copy than to rebuild for 40 min.
    rm -rf "${BACKUP_DIR}.tmp"
    mkdir -p "${BACKUP_DIR}.tmp"
    cp -a "${STACK_DIR}/bin" "${BACKUP_DIR}.tmp/bin"
    printf '%s\n' "${current}" > "${BACKUP_DIR}.tmp/version"
    rm -rf "${BACKUP_DIR}"
    mv "${BACKUP_DIR}.tmp" "${BACKUP_DIR}"
}

restore_build() {
    [ -d "${BACKUP_DIR}/bin" ] || return 1
    rm -rf "${STACK_DIR}/bin"
    cp -a "${BACKUP_DIR}/bin" "${STACK_DIR}/bin"
    if [ -f "${BACKUP_DIR}/version" ]; then
        cp -a "${BACKUP_DIR}/version" "${VERSION_FILE}"
    fi
    return 0
}

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

log "keeping ${current} in ${BACKUP_DIR} in case the new build misbehaves"
backup_build

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
    if ! wait_swap 60; then
        warn "llama-swap did not come back on ${target} - rolling back to ${current}"
        restore_build || die "no backup in ${BACKUP_DIR}; check journalctl -u llama-swap"
        systemctl restart llama-swap.service
        wait_swap 60 || die "llama-swap did not come back even on ${current}"
        die "rolled back to ${current}; the ${target} build does not start, see journalctl -u llama-swap"
    fi
    log "llama-swap healthy on ${target}"
fi

# The functional gate. It runs through the router on purpose: the router is what
# injects reasoning_budget_tokens and the tool-turn temperature floor, so verifying
# against :10001 would test a configuration no client ever uses.
if ! curl -fsS --max-time 5 "${ROUTER_URL}/health" >/dev/null 2>&1; then
    warn "router on ${ROUTER_URL} is not reachable: cannot verify tool calling"
    die "build ${target} is installed but unverified; check systemctl status llm-router"
fi

log "verifying tool calling on ${target} (${VERIFY_CASES}; ~2 min, the MoE is loading)"
if ! verify_tools; then
    warn "tool calling regressed on ${target} - retrying once before rolling back"
    if ! verify_tools; then
        warn "still failing on ${target}: restoring ${current} from ${BACKUP_DIR}"
        restore_build || die "no backup in ${BACKUP_DIR}; cannot roll back automatically"
        systemctl restart llama-swap.service || die "restart failed after rollback"
        wait_swap 60 || die "llama-swap did not come back after the rollback"
        if verify_tools; then
            log "rolled back to ${current}: tool calling works again"
            die "${target} broke tool calling and was rolled back (see journal for the battery output)"
        fi
        die "rolled back to ${current} and tool calling STILL fails: the build is not the cause"
    fi
fi

log "tool calling ok on ${target}"
log "done"
