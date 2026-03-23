# syntax=docker/dockerfile:1
# MicroAgent — Agentic Microscopy Image Analysis
# Multi-stage build with CPU and GPU targets.
#
# Build CPU image (default):
#   docker build -t microagent .
#   docker build --target runtime-cpu -t microagent:cpu .
#
# Build GPU image:
#   docker build --target runtime-gpu -t microagent:gpu .
#
# Run:
#   docker run --rm -v $(pwd)/data:/data microagent segment /data/images

# ── Stage 1: builder (shared) ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/

# Install all optional deps into a dedicated venv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache '.[all]'

# ── Stage 2a: CPU runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime-cpu

LABEL maintainer="MicroAgent Contributors"
LABEL description="Agentic microscopy image analysis tool"
LABEL license="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /data

ENTRYPOINT ["microagent"]
CMD ["--help"]

# ── Stage 2b: GPU runtime (CUDA 12.1) ────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime-gpu

LABEL maintainer="MicroAgent Contributors"
LABEL description="Agentic microscopy image analysis tool (GPU)"
LABEL license="Apache-2.0"
LABEL cuda.version="12.1"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /data

ENTRYPOINT ["microagent"]
CMD ["--help"]

# ── Default target is CPU ─────────────────────────────────────────────────────
FROM runtime-cpu AS runtime
