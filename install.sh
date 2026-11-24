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
#   sudo ./install.sh --dry-run        # print actions, change nothing
#
# Environment overrides:
#   LLAMA_CPP_TAG=v1.0.0  LLAMA_SWAP_VERSION=v275  STACK_DIR=/opt/cheap-llm-stack
#
set -euo pipefail

STACK_DIR="${STACK_DIR:-/opt/cheap-llm-stack}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${STACK_DIR}/llama.cpp"
MODELS_DIR="${STACK_DIR}/models"
BIN_DIR="${STACK_DIR}/bin"
LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-v1.0.0}"
LLAMA_SWAP_VERSION="${LLAMA_SWAP_VERSION:-v275}"
HF_BASE="https://huggingface.co"

SKIP_BUILD=0
SKIP_MODELS=0
SKIP_APT=0
DO_TUNE=0
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --skip-build)  SKIP_BUILD=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        --no-apt)      SKIP_APT=1 ;;
        --tune)        DO_TUNE=1 ;;
        --dry-run)     DRY_RUN=1 ;;
        -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
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
# 2. llama.cpp (CPU only, armv8.2-a + dotprod + fp16 — this SoC has NO i8mm)
# ---------------------------------------------------------------------------
build_llama() {
    local tag="$1"
    log "Building llama.cpp ${tag} (CPU only, A76/A55 optimised)..."
    run rm -rf "${LLAMA_DIR}"
    run git clone --depth 1 --branch "${tag}" \
        https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
    run cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_OPENMP=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_HIP=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON
    run cmake --build "${LLAMA_DIR}/build" --config Release -j"$(nproc)"
    run install -m 0755 "${LLAMA_DIR}/build/bin/llama-server" "${BIN_DIR}/llama-server"
    run install -m 0755 "${LLAMA_DIR}/build/bin/llama-bench"  "${BIN_DIR}/llama-bench"
    [ -x "${LLAMA_DIR}/build/bin/llama-cli" ] && \
        run install -m 0755 "${LLAMA_DIR}/build/bin/llama-cli" "${BIN_DIR}/llama-cli"
    run sh -c "echo '${tag}' > '${STACK_DIR}/.llama-cpp-version'"
}

if [ "${SKIP_BUILD}" = 0 ]; then
    installed_tag="$(cat "${STACK_DIR}/.llama-cpp-version" 2>/dev/null || echo none)"
    if [ -x "${BIN_DIR}/llama-server" ] && [ "${installed_tag}" = "${LLAMA_CPP_TAG}" ]; then
        log "llama.cpp ${LLAMA_CPP_TAG} already installed — skipping build"
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
if [ ! -x "${BIN_DIR}/llama-swap" ] || [ "$("${BIN_DIR}/llama-swap" --version 2>/dev/null | grep -oE 'v?[0-9]+' | head -1)" != "${LLAMA_SWAP_VERSION#v}" ]; then
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

if [ "${SKIP_MODELS}" = 0 ]; then
    log "Downloading models into ${MODELS_DIR} (long step, resumes if interrupted)..."

    # mandatory: always-on embedding model
    fetch "Qwen/Qwen3-Embedding-0.6B-GGUF" "Qwen3-Embedding-0.6B-Q8_0.gguf"

    # fast tier / classifier (resident, tiny)
    fetch "Qwen/Qwen3-0.6B-GGUF" "Qwen3-0.6B-Q8_0.gguf"

    # default workhorse: text + vision
    fetch "unsloth/Qwen3.5-2B-GGUF" "Qwen3.5-2B-UD-Q4_K_XL.gguf"
    fetch "unsloth/Qwen3.5-2B-GGUF" "mmproj-F16.gguf" "mmproj-Qwen3.5-2B-F16.gguf"

    # quality escalation: text + vision
    fetch "unsloth/Qwen3.5-4B-GGUF" "Qwen3.5-4B-UD-Q4_K_XL.gguf"
    fetch "unsloth/Qwen3.5-4B-GGUF" "mmproj-F16.gguf" "mmproj-Qwen3.5-4B-F16.gguf"

    # vision specialist: OCR / documents / screenshots
    fetch "unsloth/Qwen3-VL-2B-Instruct-GGUF" "Qwen3-VL-2B-Instruct-UD-Q4_K_XL.gguf"
    fetch "unsloth/Qwen3-VL-2B-Instruct-GGUF" "mmproj-F16.gguf" "mmproj-Qwen3-VL-2B-F16.gguf"
else
    log "Skipping model downloads (--skip-models)"
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
run systemctl enable llama-swap.service llm-router.service llama-swap-update.timer
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
