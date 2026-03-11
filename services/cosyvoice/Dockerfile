# ──────────────────────────────────────────────
# CosyVoice3 Microservice Dockerfile
# ──────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip python3-setuptools \
    ffmpeg libsndfile1 sox curl git build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    "pip>=24.0,<25" "setuptools<70" wheel

# PyTorch CUDA 12.1
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# CosyVoice3 from source (patched deps) + Matcha-TTS submodule
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && cd /opt/CosyVoice \
    && git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git third_party/Matcha-TTS \
    && sed -i 's/grpcio==1.57.0/grpcio>=1.60/' requirements.txt \
    && sed -i 's/grpcio-tools==1.57.0/grpcio-tools>=1.60/' requirements.txt \
    && sed -i '/^tensorrt/d' requirements.txt \
    && sed -i '/^openai-whisper/d' requirements.txt \
    && pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && pip install --no-cache-dir --break-system-packages --no-build-isolation openai-whisper==20231117
ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# FastAPI + uvicorn
RUN pip install --no-cache-dir --break-system-packages fastapi uvicorn[standard]

COPY server.py ./server.py
COPY default_prompt.wav ./default_prompt.wav

EXPOSE 9880

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9880", "--log-level", "info"]
