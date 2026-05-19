# ── CVLab Docker Image ─────────────────────────────────────
# Multi-stage build for minimal production image

# ── Stage 1: Build ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY cvlab/ cvlab/

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir -e . && \
    uv pip install --no-cache-dir pytest

# ── Stage 2: Runtime ──────────────────────────────────────
FROM python:3.12-slim

WORKDIR /workspace

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CVLAB_LANG=en

# Create non-root user
RUN groupadd -r cvlab && useradd -r -g cvlab -m cvlab && \
    chmod 755 /workspace

USER cvlab

ENTRYPOINT ["cvlab"]
CMD ["help"]
