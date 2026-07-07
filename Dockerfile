# Multi-stage build combining llama.cpp server and HF Trainer + DeepSpeed + PEFT
# into a single container: self-llamolotl
#
# No `# syntax=docker/dockerfile:N` directive — the rootless buildkitd that
# runs this in CI hits TLS issues when pulling the frontend image from
# docker.io (frontend image resolution uses a different containerd code
# path than regular FROM-line pulls and trips on CA propagation through
# rootlesskit's user namespace). Using BuildKit's built-in dockerfile
# frontend (selected via `buildctl build --frontend dockerfile.v0`) avoids
# the pull entirely and supports every directive used below.
#

# ─── Build Arguments ────────────────────────────────────────────────────────
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=13.1.0
ARG CUDA_SHORT=130
ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.10.0
ARG TORCH_CUDA_ARCH_LIST="8.9+PTX"
ARG TARGETARCH=amd64

# Base-image registry prefix. Default `docker.io/<path>` so this Dockerfile
# builds anywhere (local laptops, CI, ephemeral dev VMs) without extra setup.
# In our CI, .gitlab-ci.yml overrides this via `--opt build-arg:NVIDIA_CUDA_IMAGE=...`
# to point at the GitLab Dependency Proxy at
# `$CI_DEPENDENCY_PROXY_DIRECT_GROUP_IMAGE_PREFIX/nvidia/cuda`. The proxy
# fetches docker.io anonymously, caches, and serves subsequent pulls from
# the local registry — bypassing docker.io rate limits and the rootless-
# buildkitd-vs-docker.io TLS quirk documented in
# `context/refs/runner-podman-debug/FINDINGS.md`.
#
# Convention for the rest of the project:
# every external base image gets its own ARG <NAME>_IMAGE=docker.io/<path>.
# See `context/refs/dockerfile-mirror-pattern.md`.
ARG NVIDIA_CUDA_IMAGE=docker.io/nvidia/cuda

# ─── Stage 1: Build llama.cpp binaries ──────────────────────────────────────
FROM ${NVIDIA_CUDA_IMAGE}:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git libssl-dev libgomp1 && \
    rm -rf /var/cache/apt/archives && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /llama-src
COPY self.llama/ .

RUN export LDFLAGS="-L/usr/local/cuda/lib64/stubs $LDFLAGS" && \
    cmake -B build \
      -DGGML_NATIVE=OFF \
      -DGGML_CUDA=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,--allow-shlib-undefined" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,--allow-shlib-undefined" \
      . && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app /app/convert && \
    cp build/bin/llama-server /app/llama-server && \
    cp build/bin/llama-bench /app/llama-bench && \
    cp build/bin/llama-quantize /app/llama-quantize && \
    find build/bin -name "*.so*" -exec cp -P {} /app/ \; && \
    cp convert_hf_to_gguf.py /app/convert/ && \
    cp convert_lora_to_gguf.py /app/convert/ && \
    cp -r gguf-py /app/convert/gguf-py

# ─── Stage 2: Build Python training environment ─────────────────────────────
# Uses CUDA 13.1 devel image matching the runtime, with PyTorch cu130 wheels.
# DeepSpeed CPU ops are pre-built here so no JIT compilation is needed at runtime.
FROM ${NVIDIA_CUDA_IMAGE}:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS python-builder

ARG PYTORCH_VERSION
ARG CUDA_SHORT
ARG PYTHON_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG TARGETARCH

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        python3.11 python3.11-dev python3.11-venv python3-pip git build-essential \
        cmake pkg-config libnccl2 libnccl-dev curl wget libopenmpi-dev && \
    rm -rf /var/cache/apt/archives && \
    rm -rf /var/lib/apt/lists/*

# Create Python venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV MAX_JOBS="8"
ENV NVCC_THREADS="2"
ENV CMAKE_BUILD_PARALLEL_LEVEL="8"
ENV TORCH_EXTENSIONS_DIR="/opt/venv/ds_extensions"

WORKDIR /workspace

# Upgrade pip and install PyTorch with CUDA wheels
RUN pip install --upgrade pip packaging setuptools wheel psutil && \
    pip install --no-cache-dir \
        torch==${PYTORCH_VERSION}+cu${CUDA_SHORT} torchvision \
        --index-url https://download.pytorch.org/whl/cu${CUDA_SHORT} && \
    pip cache purge

# Install flash-attn (try pip install first, fallback to manual download)
RUN pip install --no-cache-dir flash-attn 2>/dev/null || \
    (echo "Installing flash-attn from prebuilt wheel..." && \
     PYTHON_CP="cp$(echo ${PYTHON_VERSION} | tr -d '.')" && \
     TORCH_MAJOR_MINOR="$(echo ${PYTORCH_VERSION} | grep -oP '^\d+\.\d+')" && \
     TORCH_TAG="torch${TORCH_MAJOR_MINOR}" && \
     case "$TARGETARCH" in \
         amd64) ARCH_TAG="x86_64" ;; \
         arm64) ARCH_TAG="aarch64" ;; \
     esac && \
     WHL_FILE="flash_attn-2.8.3+cu${CUDA_SHORT}${TORCH_TAG}-${PYTHON_CP}-${PYTHON_CP}-linux_${ARCH_TAG}.whl" && \
     wget -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/${WHL_FILE}" 2>/dev/null && \
     pip install --no-cache-dir "${WHL_FILE}" && \
     rm -f "${WHL_FILE}") || \
    echo "Note: flash-attn not available, will use standard attention" && \
    pip cache purge

# Core training stack: HF Trainer + DeepSpeed + PEFT + TRL
RUN pip install --no-cache-dir \
        transformers peft trl accelerate datasets \
        bitsandbytes mpi4py \
        fastapi "uvicorn[standard]" aiofiles gguf \
        pyyaml scipy sentencepiece protobuf aiohttp && \
    pip cache purge

# Heretic: abliteration / censorship removal tool for research

# DeepSpeed: install with CUDA check skipped (13.1 system vs 13.0 torch — minor,
# forward-compatible). Pre-build CPU Adam op so no JIT compilation needed at runtime.
# TORCH_EXTENSIONS_DIR is set to /opt/venv/ds_extensions so the compiled .so
# survives the multi-stage COPY into the runtime image.
RUN DS_SKIP_CUDA_CHECK=1 pip install --no-cache-dir deepspeed && \
    DS_SKIP_CUDA_CHECK=1 DS_ACCELERATOR=cuda python3 -c "\
from deepspeed.ops.op_builder import CPUAdamBuilder; \
b = CPUAdamBuilder(); \
b.load(verbose=True); \
print('CPU Adam built OK')" && \
    ls -la /opt/venv/ds_extensions/ && \
    pip cache purge

# Git config for model downloads
RUN git config --global credential.helper store

# ─── Stage 3: Runtime image ────────────────────────────────────────────────
FROM ${NVIDIA_CUDA_IMAGE}:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl supervisor python3.11 python3.11-dev build-essential vim nano libopenmpi-dev && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/cache/apt/archives && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* /var/tmp/*

# Create supervisor log directory
RUN mkdir -p /var/log/supervisor

# llama.cpp artifacts from builder (binaries + shared libraries + conversion scripts)
COPY --from=llama-builder /app/ /app/

# Python venv from builder
COPY --from=python-builder /opt/venv /opt/venv

ENV PATH="/app:/opt/venv/bin:${PATH}"
ENV LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64
ENV LLAMA_ARG_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV TORCH_EXTENSIONS_DIR="/opt/venv/ds_extensions"
ENV DS_SKIP_CUDA_CHECK=1
ENV HF_DATASETS_CACHE="/workspace/cache/hf-datasets"
ENV HF_HOME="/workspace/cache/hf-hub"
ENV TOKENIZED_DATASETS="/workspace/cache/tokenized-datasets"

RUN mkdir -p /workspace/cache/hf-datasets /workspace/cache/hf-hub /workspace/cache/tokenized-datasets \
    /workspace/training/configs /workspace/training/outputs /workspace/training/logs 

# supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/llamolotl.conf

# Fixed chat templates (work around llama.cpp Jinja |items bug on string arguments)
COPY chat-templates/ /app/chat-templates/

# llama-server wrapper (reads /app/llama-server.args for dynamic LoRA flags)
COPY llama-server-wrapper.sh /app/llama-server-wrapper.sh
RUN chmod +x /app/llama-server-wrapper.sh

# entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Training API, scripts, and DeepSpeed configs
COPY api/ /workspace/training/api/
COPY deepspeed_configs/ /workspace/training/deepspeed_configs/


EXPOSE 8093

WORKDIR /workspace/training

# Health check for llama-server
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
