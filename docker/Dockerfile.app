# ──────────────────────────────────────────────
# ReflexTTS Application Dockerfile
# ──────────────────────────────────────────────
# Multi-stage build: dependencies → application

# ── Stage 1: Base with Python deps ───────────
FROM python:3.12-slim AS base

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# ── Stage 2: Application ────────────────────
FROM base AS app

COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8080

CMD ["uvicorn", "src.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--log-level", "info"]
