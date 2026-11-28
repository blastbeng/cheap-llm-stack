#!/usr/bin/env bash
#
# install.sh - self-contained installer for cheap-llm-stack
#              Orange Pi Zero 3W (Allwinner A733: 2x A76 + 6x A55, 12 GB, aarch64)
#
# It does everything:
#   1. installs build/runtime dependencies
#   2. builds llama.cpp for this exact CPU (CPU-only: no CUDA/Vulkan/NPU)
#   3. installs llama-swap (aarch64 release binary)
#   4. downloads the GGUF models + vision projectors (resumable)
#   5. installs config, the llm-router and the systemd units/timer
#   6. optionally applies OS-level tuning (see scripts/tune.sh)
#
# Usage (on the board, passwordless sudo):
#   sudo ./install.sh                  # full install
#   sudo ./install.sh --skip-models    # reinstall binaries/config only
#   sudo ./install.sh --skip-build     # reuse the existing llama.cpp build
#   sudo ./install.sh --tune           # also apply sysctl/IO/CPU tuning
#   sudo ./install.sh --prune-models   # delete GGUFs that are not part of the stack
#   sudo ./install.sh --with-quality   # also fetch the opt-in Nemotron quality tier
#   sudo ./install.sh --dry-run        # print actions, change nothing
#
# Environment overrides:
#   LLAMA_CPP_TAG=v1.0.0  LLAMA_SWAP_VERSION=v275  STACK_DIR=/opt/cheap-llm-stack
#
# LLAMA_CPP_TAG names the exact ref to build (a tag or a 40-char sha) and is the
# only way to force a rebuild of a specific revision. Leave it unset: an existing
# build that starts is kept, whatever it is. The weekly updater moves this board
# onto the tip of master, and `install.sh` for a config change must not quietly
# rebuild v1.0.0 over it.
#
set -euo pipefail

STACK_DIR="${STACK_DIR:-/opt/cheap-llm-stack}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${STACK_DIR}/llama.cpp"
MODELS_DIR="${STACK_DIR}/models"
BIN_DIR="${STACK_DIR}/bin"
TAG_EXPLICIT=0
if [ -n "${LLAMA_CPP_TAG:-}" ]; then
    TAG_EXPLICIT=1
else
    LLAMA_CPP_TAG="v1.0.0"
fi
LLAMA_SWAP_VERSION="${LLAMA_SWAP_VERSION:-v275}"
HF_BASE="https://huggingface.co"

SKIP_BUILD=0
SKIP_MODELS=0
SKIP_APT=0
DO_TUNE=0
DRY_RUN=0
PRUNE_MODELS=0
WITH_QUALITY=0
for arg in "$@"; do
    case "${arg}" in
        --skip-build)   SKIP_BUILD=1 ;;
        --skip-models)  SKIP_MODELS=1 ;;
        --no-apt)       SKIP_APT=1 ;;
        --tune)         DO_TUNE=1 ;;
        --prune-models) PRUNE_MODELS=1 ;;
        --with-quality) WITH_QUALITY=1 ;;
        --dry-run)      DRY_RUN=1 ;;
        -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [ "${DRY_RUN}" = 1 ]; then printf '  + %s\n' "$*"; else "$@"; fi; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo ./install.sh)"

ARCH="$(uname -m)"
[ "${ARCH}" = "aarch64" ] || warn "architecture is ${ARCH}; the tuning below assumes aarch64/A733"

# Refuse to install on top of a half-built tree with unknown state
run mkdir -p "${STACK_DIR}" "${BIN_DIR}" "${MODELS_DIR}"

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
if [ "${SKIP_APT}" = 1 ]; then
    log "Skipping apt (--no-apt)"
else
log "Installing build and runtime dependencies..."
export DEBIAN_FRONTEND=noninteractive
run apt-get update -y
run apt-get install -y --no-install-recommends \
    git cmake build-essential ninja-build pkg-config libgomp1 \
    ca-certificates curl wget tar xz-utils jq procps psmisc \
    python3 sqlite3 libcurl4-openssl-dev
fi

# ---------------------------------------------------------------------------
# 2. llama.cpp (CPU only — this SoC has no i8mm/SVE and no usable NPU)
# ---------------------------------------------------------------------------
# Measured on this board (llama-bench, LFM2-8B-A1B Q4_0, q8_0 KV, -fa on):
#
#   -march=armv8.2-a+dotprod+fp16   t2: pp512 22.2  tg96 11.7 | t8: pp512 35.0  tg96 9.8
#   GGML_NATIVE=ON (cortex-a76/55)  t2: pp512 22.1  tg96 11.7 | t8: pp512 34.8  tg96 9.8
#   GGML_CPU_KLEIDIAI=ON            t2: pp512 21.8  tg96 11.7 | t8: pp512 33.0  tg96 9.7
#
# So: the explicit arch flag is kept (as fast as -mcpu=native and independent of
# the toolchain's idea of "native"), and KleidiAI is *not* used — its int8
# micro-kernels need i8mm to beat ggml's own repack, and A76 here has none.
# ggml's runtime repack stays enabled: it is worth ~+24% prompt processing.
LLAMA_BUILD_DIR="${STACK_DIR}/.llama-build"
LLAMA_CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_OPENMP=ON
    -DGGML_NATIVE=OFF
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
    -DGGML_CUDA=OFF
    -DGGML_VULKAN=OFF
    -DGGML_HIP=OFF
    -DLLAMA_CURL=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=ON
    # v1.0.0 links the tools against shared ggml/llama libraries and resolves
    # them through RUNPATH, which defaults to the *build* directory. Pinning it
    # to $ORIGIN makes the binaries run from ${BIN_DIR} with their libraries
    # copied next to them, and keeps the copy portable.
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    -DCMAKE_INSTALL_RPATH=\$ORIGIN
)

build_llama() {
    local tag="$1"
    log "Building llama.cpp ${tag} (CPU only, A76/A55: armv8.2-a+dotprod+fp16)..."
    run rm -rf "${LLAMA_DIR}"
    if [[ "${tag}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        # A commit sha, not a tag: `git clone --branch` cannot name one. The
        # weekly updater pins a sha (the tip of master it read from ls-remote) so
        # the build it verifies is the commit it tested, even if master moves
        # during the 40 minutes it takes to compile here.
        run git clone --depth 1 \
            https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
        run git -C "${LLAMA_DIR}" fetch --depth 1 origin "${tag}"
        run git -C "${LLAMA_DIR}" checkout --detach FETCH_HEAD
    else
        run git clone --depth 1 --branch "${tag}" \
            https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
    fi
    # Out-of-tree build: the clone stays pristine, and no leftover root-owned
    # CMake cache can wedge a later rebuild.
    run rm -rf "${LLAMA_BUILD_DIR}"
    run mkdir -p "${LLAMA_BUILD_DIR}"
    run cmake -S "${LLAMA_DIR}" -B "${LLAMA_BUILD_DIR}" -G Ninja \
        "${LLAMA_CMAKE_FLAGS[@]}"
    run cmake --build "${LLAMA_BUILD_DIR}" --config Release -j"$(nproc)"
    # The shared libraries are part of the install: without them the binaries
    # die with "libllama-server-impl.so: cannot open shared object file".
    if compgen -G "${LLAMA_BUILD_DIR}/bin/*.so*" >/dev/null; then
        run sh -c "cp -a '${LLAMA_BUILD_DIR}/bin/'*.so* '${BIN_DIR}/'"
    fi
    run install -m 0755 "${LLAMA_BUILD_DIR}/bin/llama-server" "${BIN_DIR}/llama-server"
    run install -m 0755 "${LLAMA_BUILD_DIR}/bin/llama-bench"  "${BIN_DIR}/llama-bench"
    [ -x "${LLAMA_BUILD_DIR}/bin/llama-cli" ] && \
        run install -m 0755 "${LLAMA_BUILD_DIR}/bin/llama-cli" "${BIN_DIR}/llama-cli"
    run sh -c "echo '${tag}' > '${STACK_DIR}/.llama-cpp-version'"
}

if [ "${SKIP_BUILD}" = 0 ]; then
    installed_tag="$(cat "${STACK_DIR}/.llama-cpp-version" 2>/dev/null || echo none)"
    # A binary that cannot start (missing shared library) is not an install.
    server_ok=0
    "${BIN_DIR}/llama-server" --version >/dev/null 2>&1 && server_ok=1
    # An explicit tag is a request for that revision. Without one, a binary that
    # starts is good enough: this board follows master weekly, so the installed
    # id is usually a sha and a comparison against the default tag would rebuild
    # (i.e. downgrade) on every config-only install.
    if [ "${server_ok}" = 1 ] && { [ "${TAG_EXPLICIT}" = 0 ] || [ "${installed_tag}" = "${LLAMA_CPP_TAG}" ]; }; then
        log "llama.cpp ${installed_tag} already installed — skipping build"
    else
        build_llama "${LLAMA_CPP_TAG}"
    fi
else
    [ -x "${BIN_DIR}/llama-server" ] || die "--skip-build given but ${BIN_DIR}/llama-server is missing"
    log "Skipping llama.cpp build (--skip-build)"
fi
"${BIN_DIR}/llama-server" --version 2>&1 | head -1 || true

# ---------------------------------------------------------------------------
# 3. llama-swap (aarch64 static-ish binary)
# ---------------------------------------------------------------------------
# llama-swap prints "version: v275 (e0274d5), ...": match the bare number so a
# re-install does not re-download 43 MB every time.
if [ ! -x "${BIN_DIR}/llama-swap" ] || [ "$("${BIN_DIR}/llama-swap" --version 2>/dev/null | grep -oE '[0-9]+' | head -1)" != "${LLAMA_SWAP_VERSION#v}" ]; then
    log "Installing llama-swap ${LLAMA_SWAP_VERSION} (aarch64)..."
    tmp="$(mktemp -d)"
    url="https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION#v}_linux_arm64.tar.gz"
    if ! curl -fsSL "${url}" -o "${tmp}/ls.tgz"; then
        rm -rf "${tmp}"
        die "could not download llama-swap from ${url}"
    fi
    tar -xzf "${tmp}/ls.tgz" -C "${tmp}"
    run install -m 0755 "$(find "${tmp}" -type f -name 'llama-swap*' ! -name '*.tgz' | head -1)" "${BIN_DIR}/llama-swap"
    rm -rf "${tmp}"
    "${BIN_DIR}/llama-swap" --version || true
else
    log "llama-swap ${LLAMA_SWAP_VERSION} already installed"
fi

# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
fetch() { # repo remote_file local_file
    local repo="$1" remote="$2" local_file="${3:-$2}"
    local out="${MODELS_DIR}/${local_file}"
    if [ -s "${out}" ]; then
        log "  present: ${local_file}"
        return 0
    fi
    log "  fetching ${repo}/${remote} -> ${local_file}"
    run curl -fL --retry 5 --retry-delay 3 --retry-all-errors -C - \
        "${HF_BASE}/${repo}/resolve/main/${remote}" -o "${out}"
}

# Four models, nothing else. The list is also the prune whitelist (--prune-models).
REQUIRED_MODELS=(
    "Qwen3-Embedding-0.6B-Q8_0.gguf"       # embedding-model    (mandatory, always on)
    "LFM3-1.7B-QAD-Q4_0.gguf"              # router             (fast tier: chitchat / one-liners)
    "Moon-Beam-8B-A1B-IQ4_XS.gguf"         # reasoner           (default + tool calling)
    "LFM3-VL-2B-QAD-Q4_0.gguf"             # vision             (images / OCR)
    "mmproj-LFM3-VL-2B-Q8_0.gguf"          # vision projector
)
# Opt-in quality tier (5.74 GiB, 2.91 tok/s): fetched only with --with-quality, and
# added to the prune whitelist only then.
QUALITY_MODELS=(
    "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"    # bolt / quality / deep
)
if [ "${WITH_QUALITY}" = 1 ]; then
    REQUIRED_MODELS+=("${QUALITY_MODELS[@]}")
fi

if [ "${SKIP_MODELS}" = 0 ]; then
    log "Downloading models into ${MODELS_DIR} (long step, resumes if interrupted)..."

    # mandatory: always-on embedding model (Qwen3-Embedding-0.6B, 1024 dims)
    fetch "Qwen/Qwen3-Embedding-0.6B-GGUF" "Qwen3-Embedding-0.6B-Q8_0.gguf"

    # fast tier: greetings, one-liners, short translations. LFM3-1.7B decodes at the
    # same 10.1 tok/s as the 8B MoE with 1.8x its prefill and 1 GB of weights, so it
    # is the whole fast tier: the 0.6B it replaced kept nothing but 0.64 GB of disk.
    fetch "LiquidAI/LFM3-1.7B-GGUF" "LFM3-1.7B-QAD-Q4_0.gguf"

    # text workhorse + tool calling: Moon-Beam-8B-A1B, bailingmoe3 7.89B total /
    # 1.3B active per token, IQ4_XS. 9/9 on the 9-case tool battery against 8/9 for
    # LFM2.5-8B-A1B Q4_0, and faster end to end (7.6-11.3 s per call vs 16.6-18.3 s)
    # because it thinks less. Same sparse shape, so the RAM envelope is unchanged.
    fetch "tablinum/Moon-Beam-8B-A1B-GGUF" "Moon-Beam-8B-A1B-IQ4_XS.gguf"

    # single vision model: descriptions, OCR, tables. Same transcript as Qwen3-VL-2B
    # in 24 s vs 113 s, at 1.25 GB instead of 2.4 GB.
    fetch "LiquidAI/LFM3-VL-2B-GGUF" "LFM3-VL-2B-QAD-Q4_0.gguf"
    fetch "LiquidAI/LFM3-VL-2B-GGUF" "mmproj-LFM3-VL-2B-Q8_0.gguf"

    # opt-in quality tier: Nemotron-3.5-Bolt-E4B (gemma4, dense per token). Measured
    # 2.91 tok/s and 5.74 GiB, so it swaps with the MoE instead of coexisting; it
    # is not the default and `auto` does not reach it (ROUTER_DEEP_TIER=0).
    if [ "${WITH_QUALITY}" = 1 ]; then
        fetch "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB" "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"
    fi
else
    log "Skipping model downloads (--skip-models)"
fi

# Remove GGUFs that are not part of this stack (old tiers, benchmark candidates).
# Off by default: run it explicitly with --prune-models.
if [ "${PRUNE_MODELS}" = 1 ]; then
    log "Pruning models that are not in use..."
    for f in "${MODELS_DIR}"/*.gguf; do
        [ -e "${f}" ] || continue
        keep=0
        for want in "${REQUIRED_MODELS[@]}"; do
            [ "$(basename "${f}")" = "${want}" ] && keep=1
        done
        if [ "${keep}" = 0 ]; then
            log "  removing $(basename "${f}") ($(du -h "${f}" | cut -f1))"
            run rm -f "${f}"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 5. Config, router, systemd
# ---------------------------------------------------------------------------
log "Deploying configuration, router and systemd units..."

# place <mode> <src> <dst> — install src at dst, but tolerate running straight
# from the installed tree (SRC_DIR == STACK_DIR), where src and dst are the same
# inode and `install` would abort with "are the same file".
place() {
    local mode="$1" src="$2" dst="$3"
    if [ "$(readlink -f "${src}")" = "$(readlink -f "${dst}" 2>/dev/null)" ]; then
        return 0
    fi
    run install -m "${mode}" "${src}" "${dst}"
}

run install -d -m 0755 "${STACK_DIR}/router" "${STACK_DIR}/scripts"
place 0644 "${SRC_DIR}/config.yaml"             "${STACK_DIR}/config.yaml"
place 0755 "${SRC_DIR}/router/llm_router.py"    "${STACK_DIR}/router/llm_router.py"
place 0755 "${SRC_DIR}/scripts/tune.sh"         "${STACK_DIR}/scripts/tune.sh"
place 0755 "${SRC_DIR}/scripts/toolcheck.py"    "${STACK_DIR}/scripts/toolcheck.py"
place 0755 "${SRC_DIR}/update-llama.cpp.sh"     "${STACK_DIR}/update-llama.cpp.sh"
place 0755 "${SRC_DIR}/benchmark.sh"            "${STACK_DIR}/benchmark.sh"
place 0755 "${SRC_DIR}/healthcheck.sh"          "${STACK_DIR}/healthcheck.sh"
place 0644 "${SRC_DIR}/README.md"               "${STACK_DIR}/README.md"

run install -m 0755 "${SRC_DIR}/systemd/llama-swap.service"        /etc/systemd/system/llama-swap.service
run install -m 0755 "${SRC_DIR}/systemd/llm-router.service"       /etc/systemd/system/llm-router.service
run install -m 0755 "${SRC_DIR}/systemd/llama-swap-update.service" /etc/systemd/system/llama-swap-update.service
run install -m 0755 "${SRC_DIR}/systemd/llama-swap-update.timer"   /etc/systemd/system/llama-swap-update.timer

if [ "${DO_TUNE}" = 1 ]; then
    log "Applying OS-level tuning..."
    run "${SRC_DIR}/scripts/tune.sh" --apply
else
    log "OS tuning not applied (add --tune, or run ${SRC_DIR}/scripts/tune.sh --check)"
fi

run systemctl daemon-reload
run systemctl enable llama-swap.service llm-router.service
# enable + start: an enabled timer is not armed until it is started or the board reboots
run systemctl enable --now llama-swap-update.timer
# clear the restart counter left behind by a previously broken ExecStart
systemctl reset-failed llama-swap.service llm-router.service 2>/dev/null || true
run systemctl restart llama-swap.service
run sleep 2
run systemctl restart llm-router.service

# ---------------------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------------------
if [ "${DRY_RUN}" = 0 ]; then
    log "Waiting for llama-swap and the router..."
    ok=0
    for _ in $(seq 1 90); do
        if curl -fsS "http://127.0.0.1:10001/v1/models" >/dev/null 2>&1 \
           && curl -fsS "http://127.0.0.1:10000/health" >/dev/null 2>&1; then ok=1; break; fi
        sleep 2
    done
    if [ "${ok}" = 1 ]; then
        log "Stack is up."
        curl -fsS "http://127.0.0.1:10000/stats" 2>/dev/null || true
        echo
    else
        warn "Stack did not answer in 180 s. Inspect: journalctl -u llama-swap -u llm-router -n 100"
    fi
fi

log "Install complete."
log "  Router (OpenAI-compatible):  http://<board>:10000/v1   model: auto"
log "  llama-swap (internal):       http://127.0.0.1:10001/v1"
log "  Health:                      curl http://127.0.0.1:10000/health"
log "  Benchmark:                   ${STACK_DIR}/benchmark.sh --all"
