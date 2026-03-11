# ──────────────────────────────────────────────
# ReflexTTS App Dockerfile (lightweight)
# ──────────────────────────────────────────────
# No CUDA needed — app only makes HTTP calls
# to vLLM, CosyVoice, and WhisperX services.

FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[dev]"

# ── Application ────────────────────
FROM base AS app

COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8081

CMD ["uvicorn", "src.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8081", \
     "--log-level", "info"]
