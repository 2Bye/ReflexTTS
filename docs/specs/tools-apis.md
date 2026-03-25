# Spec: Tools / APIs

> Контракты, ошибки, timeout, side effects, защита.

---

## 1. VLLMClient — LLM Inference

**Файл:** `src/inference/vllm_client.py`
**Модель:** Qwen3-8B-Instruct AWQ 4-bit
**Протокол:** OpenAI-compatible API (AsyncOpenAI)

### Контракт

| Метод | Signature | Return | Used by |
|-------|-----------|--------|---------|
| `chat()` | `(system_prompt, user_message, temperature?, max_tokens?)` | `str` | — |
| `chat_json()` | `(system_prompt, user_message, response_model: type[T])` | `T` (Pydantic) | Director, Critic |
| `health_check()` | `()` | `bool` | ModelRegistry |
| `close()` | `()` | `None` | Pipeline cleanup |

### Error handling

| Ошибка | Обработка | Retry |
|--------|----------|-------|
| `APIConnectionError` | Exponential backoff: `2^attempt` sec | 5× |
| `APITimeoutError` | Exponential backoff | 5× |
| `APIStatusError` | Immediate fail → `VLLMResponseError` | ❌ |
| `JSONDecodeError` | 3-step fallback: strip `<think>` → parse → brace extract | ❌ |
| Empty response | `VLLMResponseError` | ❌ |

### Configuration

| Параметр | Default | Env var |
|----------|---------|---------|
| `base_url` | `http://localhost:8000/v1` | `VLLM_BASE_URL` |
| `model_name` | `Qwen/Qwen3-8B-AWQ` | `VLLM_MODEL_NAME` |
| `max_tokens` | 4096 | `VLLM_MAX_TOKENS` |
| `temperature` | 0.1 | `VLLM_TEMPERATURE` |
| `timeout_seconds` | 300 | `VLLM_TIMEOUT_SECONDS` |
| `max_retries` | 5 | `VLLM_MAX_RETRIES` |

### Side effects
- **GPU**: ~5 GB VRAM allocated continuously
- **Network**: HTTP requests to vLLM server
- **No persistent storage** — stateless per request

---

## 2. TTSClient — Speech Synthesis

**Файл:** `src/inference/tts_client.py`
**Модель:** CosyVoice3 0.5B (Fun-CosyVoice3-0.5B)
**Протокол:** HTTP REST (httpx)

### Контракт

| Метод | Signature | Return | Used by |
|-------|-----------|--------|---------|
| `synthesize()` | `(text, voice_id, instruct?)` | `AudioResult(waveform, sample_rate)` | Actor, Editor |
| `clone_voice()` | `(text, ref_audio, ref_text)` | `AudioResult` | — (disabled in PoC) |
| `load_model()` | `()` | `None` | Pipeline startup |
| `health_check()` | `()` | `bool` | ModelRegistry |
| `close()` | `()` | `None` | Pipeline cleanup |

### Voice mapping

| voice_id | CosyVoice speaker | Language |
|----------|-------------------|----------|
| `speaker_1` | 中文女 | Chinese |
| `speaker_2` | 中文男 | Chinese |
| `speaker_3` | 英文女 | English |

### Error handling

| Ошибка | Причина | Обработка |
|--------|---------|----------|
| `httpx.TimeoutException` | CosyVoice GPU overload | Pipeline fail |
| `AssertionError` (instruct token) | Missing `<\|endofprompt\|>` | Auto-appended in server.py |
| HTTP 500 | CosyVoice internal error | Pipeline fail |

### Configuration

| Параметр | Default | Env var |
|----------|---------|---------|
| `base_url` | `http://localhost:9880` | `COSYVOICE_BASE_URL` |
| `sample_rate` | 24000 | `COSYVOICE_SAMPLE_RATE` |

### Side effects
- **GPU**: ~2 GB VRAM
- **Audio output**: WAV bytes returned in response (not written to disk)

---

## 3. ASRClient — Speech Recognition

**Файл:** `src/inference/asr_client.py`
**Модель:** WhisperX large-v3 + Wav2Vec2 (forced alignment)
**Протокол:** HTTP REST (httpx)

### Контракт

| Метод | Signature | Return | Used by |
|-------|-----------|--------|---------|
| `transcribe()` | `(audio_bytes, sample_rate)` | `TranscriptionResult` | Critic |
| `load_model()` | `()` | `None` | Pipeline startup |
| `health_check()` | `()` | `bool` | ModelRegistry |
| `close()` | `()` | `None` | Pipeline cleanup |

### TranscriptionResult

```python
TranscriptionResult(
    text: str,                    # Full transcript
    word_timestamps: list[WordTimestamp],  # Per-word timing
    language: str                 # Detected language
)

WordTimestamp(
    word: str,       # "Hello,"
    start_ms: float, # 240.0
    end_ms: float,   # 680.0
    score: float     # 0.95 (confidence)
)
```

### Configuration

| Параметр | Default | Env var |
|----------|---------|---------|
| `base_url` | `http://localhost:9881` | `WHISPERX_BASE_URL` |
| `model_name` | `large-v3` | `WHISPERX_MODEL_NAME` |
| `device` | `cuda` | `WHISPERX_DEVICE` |
| `compute_type` | `float16` | `WHISPERX_COMPUTE_TYPE` |

### Side effects
- **GPU**: ~3 GB VRAM
- **No persistent storage**

---

## 4. Security APIs (internal)

### Input Sanitizer

| Метод | Signature | Return |
|-------|-----------|--------|
| `sanitize_input()` | `(text, max_length?, strict?)` | `SanitizeResult(is_safe, sanitized_text, reason, matched_patterns)` |
| `strip_control_chars()` | `(text)` | `str` |

**10 patterns**: ignore_previous, disregard_previous, role_override, act_as, pretend, system_prompt_inject, chat_template_inject, code_block_inject, xss_attempt, template_inject

### PII Masker

| Метод | Signature | Return |
|-------|-----------|--------|
| `mask_pii()` | `(text)` | `PIIResult(masked_text, pii_count, pii_types)` |

**Types**: email, phone, card, passport, INN, IP address

### Voice Whitelist

| Метод | Signature | Return |
|-------|-----------|--------|
| `validate_voice()` | `(voice_id, config)` | `None` (raises `VoiceNotAllowedError`) |

---

## 5. REST API

| Endpoint | Method | Input | Output | Status |
|----------|--------|-------|--------|--------|
| `/health` | GET | — | `{"status": "ok"}` | 200 |
| `/voices` | GET | — | `{"voices": [...]}` | 200 |
| `/synthesize` | POST | `SynthesizeRequest` | `SynthesizeResponse` | 202 |
| `/session/{id}/status` | GET | — | `SessionStatus` | 200/404 |
| `/session/{id}/audio` | GET | — | WAV bytes | 200/404/409 |
| `/metrics` | GET | — | Prometheus text | 200 |
| `/ws/{session_id}` | WS | — | Agent log stream | — |

### Error codes

| Code | Cause |
|------|-------|
| 400 | Invalid input, prompt injection, disallowed voice |
| 404 | Session not found |
| 409 | Audio not ready (still processing) |
| 503 | Pipeline busy (another synthesis in progress) |
