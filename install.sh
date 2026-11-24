#!/usr/bin/env bash
#
# install.sh - Self-contained installer for the cheap-llm-stack on
#              Orange Pi Zero 3W (Allwinner A733, aarch64, Armbian).
#
# It does everything:
#   1. Installs build/runtime dependencies.
#   2. Builds llama.cpp (CPU only: no CUDA/NPU/Vulkan).
#   3. Installs llama-swap (aarch64 binary).
#   4. Downloads all GGUF models + vision projectors.
#   5. Installs and starts the systemd services.
#
# Usage (on the board, as a user with passwordless sudo):
#   sudo ./install.sh
#
set -euo pipefail

STACK_DIR="/opt/cheap-llm-stack"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${STACK_DIR}/llama.cpp"
MODELS_DIR="${STACK_DIR}/models"
BIN_DIR="${STACK_DIR}/bin"
LLAMA_SWAP_VERSION="v0.15.0"
LLAMA_CPP_TAG="v1.0.0"

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo ./install.sh)"
[ "$(uname -m)" = "aarch64" ] || warn "not aarch64; build flags assume the A733"

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
log "Installing build and runtime dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config \
    libcurl4-openssl-dev ca-certificates curl wget tar \
    python3 python3-venv sqlite3

# ---------------------------------------------------------------------------
# 2. llama.cpp (CPU only)
# ---------------------------------------------------------------------------
mkdir -p "${STACK_DIR}" "${BIN_DIR}" "${MODELS_DIR}"

if [ ! -x "${LLAMA_DIR}/build/bin/llama-server" ]; then
    log "Building llama.cpp ${LLAMA_CPP_TAG} (CPU only, -t 2 target A76)..."
    rm -rf "${LLAMA_DIR}"
    git clone --depth 1 --branch "${LLAMA_CPP_TAG}" \
        https://github.com/ggerganov/llama.cpp.git "${LLAMA_DIR}"
    cd "${LLAMA_DIR}"
    # CPU-only build. No GGML_CUDA / GGML_VULKAN / NPU backends.
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_SERVER=ON
    cmake --build build --config Release -j"$(nproc)"
    cd "${SRC_DIR}"
    echo "${LLAMA_CPP_TAG}" > "${STACK_DIR}/.llama-cpp-version"
fi

# ---------------------------------------------------------------------------
# 3. llama-swap (aarch64)
# ---------------------------------------------------------------------------
if [ ! -x "${BIN_DIR}/llama-swap" ]; then
    log "Installing llama-swap ${LLAMA_SWAP_VERSION} (aarch64)..."
    tmp="$(mktemp -d)"
    url="https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION#v}_linux_arm64.tar.gz"
    if ! curl -fsSL "${url}" -o "${tmp}/ls.tgz"; then
        # Fallback: try the generic asset name.
        curl -fsSL "https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/llama-swap_linux_arm64.tar.gz" -o "${tmp}/ls.tgz" \
            || die "could not download llama-swap ${LLAMA_SWAP_VERSION}"
    fi
    tar -xzf "${tmp}/ls.tgz" -C "${tmp}"
    install -m 0755 "$(find "${tmp}" -type f -name 'llama-swap*' | head -1)" "${BIN_DIR}/llama-swap"
    rm -rf "${tmp}"
    "${BIN_DIR}/llama-swap" --version || true
fi

# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
log "Downloading models (this is the long step; resumes if interrupted)..."
HF_BASE="https://huggingface.co"
fetch() { # repo path -> models_dir
    local repo="$1" file="$2" out="${MODELS_DIR}/$2"
    if [ -s "${out}" ]; then
        log "  exists: ${file}"
        return
    fi
    log "  fetching ${repo}/${file}"
    curl -fL --retry 5 --retry-delay 3 -C - \
        "${HF_BASE}/${repo}/resolve/main/${file}" -o "${out}"
}

# Embedding (mandatory, always online)
fetch "Qwen/Qwen3-Embedding-0.6B-GGUF" "Qwen3-Embedding-0.6B-Q8_0.gguf"

# 0.8B (vision) - Q8_K_XL for quality
fetch "unsloth/Qwen3.5-0.8B-GGUF" "Qwen3.5-0.8B-UD-Q8_K_XL.gguf"
fetch "unsloth/Qwen3.5-0.8B-GGUF" "Qwen3.5-0.8B-mmproj-F16.gguf"

# 2B (primary chat + reasoning + vision) - Q4_K_XL
fetch "unsloth/Qwen3.5-2B-GGUF" "Qwen3.5-2B-UD-Q4_K_XL.gguf"
fetch "unsloth/Qwen3.5-2B-GGUF" "Qwen3.5-2B-mmproj-F16.gguf"

# 4B (quality, hot-swap) - Q4_K_XL
fetch "unsloth/Qwen3.5-4B-GGUF" "Qwen3.5-4B-UD-Q4_K_XL.gguf"
fetch "unsloth/Qwen3.5-4B-GGUF" "Qwen3.5-4B-mmproj-F16.gguf"

# ---------------------------------------------------------------------------
# 5. Deploy config + systemd units
# ---------------------------------------------------------------------------
log "Deploying configuration and systemd units..."
install -m 0644 "${SRC_DIR}/config.yaml" "${STACK_DIR}/config.yaml"
install -m 0755 "${SRC_DIR}/update-llama.cpp.sh" "${STACK_DIR}/update-llama.cpp.sh"
install -m 0755 "${SRC_DIR}/benchmark.sh" "${STACK_DIR}/benchmark.sh"
install -m 0755 "${SRC_DIR}/healthcheck.sh" "${STACK_DIR}/healthcheck.sh"

install -m 0644 "${SRC_DIR}/systemd/llama-swap.service" /etc/systemd/system/llama-swap.service
install -m 0644 "${SRC_DIR}/systemd/llama-swap-update.service" /etc/systemd/system/llama-swap-update.service
install -m 0644 "${SRC_DIR}/systemd/llama-swap-update.timer" /etc/systemd/system/llama-swap-update.timer

systemctl daemon-reload
systemctl enable --now llama-swap.service
systemctl enable --now llama-swap-update.timer

# ---------------------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------------------
log "Waiting for llama-swap to come up..."
for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:10001/v1/models" >/dev/null 2>&1; then
        log "llama-swap is up at http://127.0.0.1:10001"
        break
    fi
    sleep 2
    [ "$i" -eq 60 ] && warn "llama-swap did not answer in 120s; check: journalctl -u llama-swap -n 100"
done

log "Install complete. Try:"
log "  curl http://127.0.0.1:10001/v1/models"
log "  curl http://127.0.0.1:10001/v1/embeddings -d '{\"model\":\"embedding-model\",\"input\":\"hello\"}'"
