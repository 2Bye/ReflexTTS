# ──────────────────────────────────────────────
# WhisperX Microservice Dockerfile
# ──────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + FFmpeg 6 (Ubuntu 22.04 ships FFmpeg 4.4)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-setuptools \
    ffmpeg libsndfile1 curl git build-essential \
    software-properties-common \
    && add-apt-repository ppa:ubuntuhandbook1/ffmpeg6 || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade \
    "pip>=24.0,<25" "setuptools<70" wheel

# PyTorch CUDA 12.4 (matches base image)
RUN pip install --no-cache-dir \
    torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# WhisperX (will use our pinned torch)
RUN pip install --no-cache-dir whisperx

# Disable torchcodec (pyannote fallback to soundfile is fine)
ENV TORCHCODEC_DISABLE=1

# FastAPI + uvicorn + audio deps
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart soundfile

COPY server.py ./server.py

EXPOSE 9881

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9881", "--log-level", "info"]
