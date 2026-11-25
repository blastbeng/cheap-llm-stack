#!/usr/bin/env bash
#
# update-llama.cpp.sh - rebuild llama.cpp when upstream has moved, then prove
# that the new build can still call tools.
#
# Run weekly by llama-swap-update.timer (Sun 03:00) or by hand:
#   sudo ./update-llama.cpp.sh                    # tip of master (default)
#   sudo LLAMA_CPP_SOURCE=release ./update-llama.cpp.sh    # newest release tag
#   sudo LLAMA_CPP_TAG=v1.0.0 ./update-llama.cpp.sh        # one exact ref
#
# Why master and not releases
# ---------------------------
# llama.cpp tags rarely: the newest release is v1.0.0 (2026-11-13), which is the
# build this board already runs, so a release tracker would compile nothing for
# months and the verification below would never run. The things this script
# exists to catch - chat-template handlers, tool parsers, ARM kernels - land on
# master, and are usually fixed there within days. Weekly master is the same risk
# as a release once there is a gate that rolls back: it buys a year of fixes and
# costs one night's compile, and a broken tip is skipped, not installed.
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
# and the unit is left failed so `systemctl --failed` shows it. The sha that
# failed goes to .llama-cpp-broken, so the next Sunday does not spend 40 minutes
# recompiling the same broken tip; when master moves again the marker no longer
# matches and the updater tries the new one.
#
set -euo pipefail

STACK_DIR="${STACK_DIR:-/opt/cheap-llm-stack}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${STACK_DIR}/.llama-cpp-version"
BROKEN_FILE="${STACK_DIR}/.llama-cpp-broken"
BACKUP_DIR="${STACK_DIR}/.build-backup"
TOOLCHECK="${SRC_DIR}/scripts/toolcheck.py"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:10000}"
# `vec` is deliberately absent: it exercises an invented identifier inside the
# arguments, which every model on this board mangles, so it warns rather than
# fails and adds 20 s without adding signal about the build.
VERIFY_CASES="${VERIFY_CASES:-tg,web,mem,wx,multi,none}"
REPO="ggml-org/llama.cpp"
REPO_URL="https://github.com/${REPO}.git"
# "master" (default) tracks the tip commit; "release" tracks the newest
# non-prerelease tag. LLAMA_CPP_TAG pins an exact ref and wins over both.
SOURCE="${LLAMA_CPP_SOURCE:-master}"

short() { printf '%s' "${1:0:9}"; }

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
    rm -rf "${STACK_DIR:?}/bin"
    cp -a "${BACKUP_DIR}/bin" "${STACK_DIR}/bin"
    if [ -f "${BACKUP_DIR}/version" ]; then
        cp -a "${BACKUP_DIR}/version" "${VERSION_FILE}"
    fi
    return 0
}

if [ -n "${LLAMA_CPP_TAG:-}" ]; then
    target="${LLAMA_CPP_TAG}"
    log "pinned to ${target} by LLAMA_CPP_TAG"
elif [ "${SOURCE}" = "release" ]; then
    # /releases/latest is one request and gives the newest non-prerelease tag;
    # the tags endpoint is rate-limited harder and can return RC candidates.
    target="$(curl -fsSL --max-time 30 "https://api.github.com/repos/${REPO}/releases/latest" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')" \
        || die "cannot query GitHub releases for ${REPO}"
elif [ "${SOURCE}" = "master" ]; then
    # ls-remote is one packet round trip, unauthenticated and unthrottled; the
    # API's /commits endpoint is rate-limited to 60/h per IP, which a board that
    # also runs other traffic can burn.
    target="$(git ls-remote "${REPO_URL}" master | awk 'NR==1{print $1}')" \
        || die "cannot resolve the tip of ${REPO} master"
    [ "${#target}" -eq 40 ] || die "cannot resolve the tip of ${REPO} master (got '${target}')"
else
    die "LLAMA_CPP_SOURCE must be 'master' or 'release' (got '${SOURCE}')"
fi

log "installed: $(short "${current}")    upstream: $(short "${target}")"
if [ "${target}" = "${current}" ]; then
    log "already up to date"
    exit 0
fi

# Upstream moved onto a commit that already failed the battery here. Rebuilding
# the same broken tip every Sunday costs 40 minutes and achieves nothing, so it
# is skipped until the tip moves on its own.
if [ -f "${BROKEN_FILE}" ] && [ "$(cat "${BROKEN_FILE}")" = "${target}" ]; then
    warn "$(short "${target}") is the commit that broke the last build; staying on $(short "${current}")"
    warn "it clears by itself when master moves. Force it with:"
    warn "  sudo LLAMA_CPP_TAG=${target} ./update-llama.cpp.sh"
    exit 0
fi

was_active=0
systemctl is-active --quiet llama-swap.service && was_active=1

log "keeping ${current} in ${BACKUP_DIR} in case the new build misbehaves"
backup_build

log "rebuilding llama.cpp $(short "${target}") (this takes ~20-40 min on this board)"
if LLAMA_CPP_TAG="${target}" "${SRC_DIR}/install.sh" --skip-models --no-apt; then
    # The version string, not the sha: on master the sha says nothing about what
    # shipped, and this is the line that ends up in `journalctl -u llama-swap-update`.
    built="$("${STACK_DIR}/bin/llama-server" --version 2>&1 | head -n1 || true)"
    log "build ok: ${built:-$(cat "${VERSION_FILE}" 2>/dev/null)}"
else
    warn "build failed - keeping the running binary ($(short "${current}"))"
    printf '%s\n' "${target}" > "${BROKEN_FILE}"
    exit 1
fi

if [ "${was_active}" = 1 ]; then
    log "restarting llama-swap"
    systemctl restart llama-swap.service
    if ! wait_swap 60; then
        warn "llama-swap did not come back on $(short "${target}") - rolling back to $(short "${current}")"
        restore_build || die "no backup in ${BACKUP_DIR}; check journalctl -u llama-swap"
        printf '%s\n' "${target}" > "${BROKEN_FILE}"
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
        printf '%s\n' "${target}" > "${BROKEN_FILE}"
        systemctl restart llama-swap.service || die "restart failed after rollback"
        wait_swap 60 || die "llama-swap did not come back after the rollback"
        if verify_tools; then
            log "rolled back to ${current}: tool calling works again"
            die "${target} broke tool calling and was rolled back (see journal for the battery output)"
        fi
        die "rolled back to ${current} and tool calling STILL fails: the build is not the cause"
    fi
fi

# Verified end to end: this build is now the known-good one, so forget the sha
# that failed before it.
rm -f "${BROKEN_FILE}"

log "tool calling ok on ${target}"
log "done"
