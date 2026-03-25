# Spec: Serving / Config

> Запуск, конфигурация, секреты, версии моделей.

---

## 1. Запуск

### Local (dev)

```bash
# Установка
pip install -e ".[dev]"

# Проверки
ruff check src/ tests/
python -m mypy src/
python -m pytest tests/unit/ -v

# Запуск API (без GPU)
python -m uvicorn src.api.app:create_app --factory --port 8080
```

### Docker Compose (GPU production)

```bash
cd docker && docker compose up -d

# Сервисы:
# vLLM:      :8055 (GPU 1)
# CosyVoice: :9880 (GPU 2)
# WhisperX:  :9881 (GPU 2)
# Redis:     :8056
# App:       :8081
```

### Healthchecks

| Сервис | Endpoint | Interval | Timeout | Start period |
|--------|----------|----------|---------|-------------|
| vLLM | `GET :8055/health` | 30s | 10s | 120s |
| CosyVoice | `GET :9880/health` | 30s | 10s | 300s |
| WhisperX | `GET :9881/health` | 30s | 10s | 180s |
| Redis | `redis-cli -p 8056 ping` | 10s | 5s | — |
| App | `GET :8081/health` | 15s | 5s | 30s |

---

## 2. Конфигурация

Вся конфигурация через **environment variables** + Pydantic Settings.
**Файл:** `src/config.py`

### Иерархия конфигурации

```
Priority (high → low):
  1. docker-compose.yml → environment:
  2. .env file (env_file:)
  3. Code defaults (Pydantic Field default)
```

### Полная таблица

| Prefix | Variable | Default | Description |
|--------|----------|---------|-------------|
| `VLLM_` | `BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| | `MODEL_NAME` | `Qwen/Qwen3-8B-AWQ` | Model identifier |
| | `MAX_TOKENS` | `4096` | Max response tokens |
| | `TEMPERATURE` | `0.1` | Sampling temperature |
| | `TIMEOUT_SECONDS` | `300` | Request timeout |
| | `MAX_RETRIES` | `5` | Connection retry count |
| `COSYVOICE_` | `BASE_URL` | `http://localhost:9880` | CosyVoice server |
| | `MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | Model path |
| | `SAMPLE_RATE` | `24000` | Output sample rate |
| `WHISPERX_` | `BASE_URL` | `http://localhost:9881` | WhisperX server |
| | `MODEL_NAME` | `large-v3` | Whisper model version |
| | `DEVICE` | `cuda` | Compute device |
| | `COMPUTE_TYPE` | `float16` | Precision |
| `SECURITY_` | `MAX_TEXT_LENGTH` | `5000` | Max input chars |
| | `MAX_RETRIES` | `5` | Pipeline retry limit |
| | `WER_THRESHOLD_FOR_HUMAN_REVIEW` | `0.15` | WER escalation |
| | `WHITELISTED_VOICES` | `["speaker_1","speaker_2","speaker_3"]` | Allowed voices |
| | `ENABLE_PII_MASKING` | `true` | PII masking toggle |
| | `ENABLE_INPUT_SANITIZATION` | `true` | Injection guard |
| `REDIS_` | `URL` | `redis://localhost:6379/0` | Redis connection |
| | `SESSION_TTL_SECONDS` | `3600` | Session expiry |
| `LOG_` | `LEVEL` | `INFO` | Log level |
| | `FORMAT` | `json` | `json` / `console` |
| | `SERVICE_NAME` | `reflex-tts` | structlog service tag |
| | `ENABLE_OTEL` | `false` | OpenTelemetry tracing |
| | `OTEL_ENDPOINT` | `http://localhost:4317` | OTel collector |
| `API_` | `HOST` | `0.0.0.0` | Server bind |
| | `PORT` | `8080` | Server port |
| | `WORKERS` | `1` | Uvicorn workers |
| | `CORS_ORIGINS` | `["http://localhost:3000","http://localhost:8080"]` | CORS |
| | `RATE_LIMIT_PER_MINUTE` | `10` | Request rate limit |

---

## 3. Секреты

| Секрет | Хранение | Описание |
|--------|---------|----------|
| `VLLM_API_KEY` | `.env` / env var | `"not-needed"` для локального vLLM |
| `REDIS_URL` | `.env` / env var | Connection string с паролем (если есть) |
| Нет облачных API ключей | — | Полностью self-hosted |

> **Важно**: `.env.example` содержит только шаблоны. Реальные `.env` файлы в `.gitignore`.

---

## 4. Версии моделей

| Модель | Версия | Source | Quantization | VRAM |
|--------|--------|--------|-------------|------|
| **Qwen3-8B-Instruct** | AWQ 4-bit | `Qwen/Qwen3-8B-AWQ` | AWQ Marlin | ~5 GB |
| **CosyVoice3** | 0.5B | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | FP16 | ~2 GB |
| **WhisperX** | large-v3 | `openai/whisper-large-v3` + Wav2Vec2 | FP16 | ~3 GB |

### GPU Requirements

| GPU | VRAM | Services | Utilization |
|-----|------|----------|-------------|
| GPU 1 (A4000 16GB) | ~5 GB used | vLLM only | 0.7 (mem util) |
| GPU 2 (A4000 16GB) | ~5 GB used | CosyVoice + WhisperX | Shared |

### Docker Image Versions

| Image | Version | Purpose |
|-------|---------|---------|
| `vllm/vllm-openai` | latest | vLLM inference server |
| `redis` | 7-alpine | Session store |
| Custom `Dockerfile.app` | — | FastAPI application |
| Custom CosyVoice Dockerfile | — | TTS service |
| Custom WhisperX Dockerfile | — | ASR service |

---

## 5. Deployment Topology

```
    ┌─────────────────────────────────────────────────────────────┐
    │                     Docker Compose                           │
    │                                                              │
    │   GPU 1                    GPU 2                 CPU         │
    │   ┌────────────────┐      ┌────────────────┐   ┌─────────┐ │
    │   │ reflex-vllm    │      │ reflex-cosyvoice│   │reflex-  │ │
    │   │ :8055          │      │ :9880           │   │app :8081│ │
    │   │ ~5GB VRAM      │      │ ~2GB VRAM       │   │         │ │
    │   └────────────────┘      ├────────────────┤   └─────────┘ │
    │                           │ reflex-whisperx │   ┌─────────┐ │
    │                           │ :9881           │   │reflex-  │ │
    │                           │ ~3GB VRAM       │   │redis    │ │
    │                           └────────────────┘   │:8056    │ │
    │                                                 └─────────┘ │
    │                                                              │
    │   network_mode: host (все сервисы)                          │
    └─────────────────────────────────────────────────────────────┘
```
