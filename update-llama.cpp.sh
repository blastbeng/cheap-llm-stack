#!/usr/bin/env bash
#
# update-llama.cpp.sh - Monthly llama.cpp rebuild (weekly/monthly).
# Fetches upstream releases and rebuilds only if a newer tag exists.
# Tag file: .llama-cpp-version in STACK_DIR.
#
set -euo pipefail

STACK_DIR="/opt/cheap-llm-stack"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${STACK_DIR}/llama.cpp"
BIN_DIR="${STACK_DIR}/bin"

log()  { printf '\033[1;32m[update]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "${LLAMA_DIR}/build" ] || die "llama.cpp not built; run install.sh first"

current=$(cat "${STACK_DIR}/.llama-cpp-version" 2>/dev/null || echo "none")
log "Current llama.cpp commit/tag: ${current}"

latest=$(curl -fsSL "https://api.github.com/repos/ggerganov/llama.cpp/tags" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['name'])")

if [ "${latest}" = "${current}" ]; then
    log "Up-to-date: ${latest}"
    exit 0
fi

log "Newer release detected: ${latest} > ${current}"
log "Rebuilding llama.cpp..."

(
    cd "${LLAMA_DIR}"
    # Stash current build state
    cp -a build/ "build.${current}.backup"
    # Fresh build from the new tag
    git fetch origin
    git checkout "${latest}"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_SERVER=ON
    cmake --build build --config Release -j"$(nproc)"
    # Mark success in better place; service monitor can use this.
    git describe --tags --exact-match >/dev/null 2>&1 && \
        echo "${latest}" > "${STACK_DIR}/.llama-cpp-version"
)

if [ $? -eq 0 ]; then
    log "llama.cpp updated to ${latest}; restart llama-swap..."
    systemctl restart llama-swap.service
else
    warn "Build failed; restored backup."
    rm -rf "${LLAMA_DIR}/build"
    mv "${LLAMA_DIR}/build.${current}.backup" "${LLAMA_DIR}/build"
    die "failed to build and deploy updated llama.cpp"
fi
