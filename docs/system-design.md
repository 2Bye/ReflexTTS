# ReflexTTS — System Design

> Архитектурный документ PoC-системы самокорректирующегося речевого синтеза.
> Версия: 1.0 · Milestone 2 · 2026-03-25

---

## 1. Ключевые архитектурные решения

### 1.1 Мульти-агентный подход (Reflection pattern)

| Решение | Обоснование |
|---------|-------------|
| **4 агента** (Director, Actor, Critic, Editor) вместо монолитного pipeline | Каждый агент — single responsibility; простота отладки, замены и масштабирования |
| **Centralized orchestrator** (LangGraph DAG) | Предсказуемый control flow, детерминированный routing, упрощённая отладка vs decentralized MAS |
| **Shared blackboard** (`GraphState`) | Передача данных между агентами через one Pydantic model; всё состояние в одном месте |
| **LLM-in-the-loop** для контроля качества | Critic Agent с двухфазной оценкой (ASR + LLM Judge) — 100% семантическая верификация |
| **Segment-level операции** | Per-segment synthesis, evaluation и repair — минимизация GPU-работы при retry |

### 1.2 Инференс-стек: полностью локальный

| Решение | Обоснование |
|---------|-------------|
| **Qwen3-8B AWQ 4-bit** через vLLM | OpenAI-compatible API, ~5 GB VRAM, JSON mode, поддержка reasoning (`<think>`) |
| **CosyVoice3 0.5B** (Flow-Matching) | Bidirectional attention для inpainting, multi-speaker, instruct mode |
| **WhisperX large-v3** + Wav2Vec2 forced alignment | Word-level timestamps с confidence scores; необходимо для точной локализации ошибок |
| **Все модели — локально** | Нет зависимости от cloud API, контроль latency, нет PII leakage к третьим сторонам |

### 1.3 Инфраструктурные решения

| Решение | Обоснование |
|---------|-------------|
| **FastAPI** (async, WebSocket) | Async pipeline + real-time streaming логов; встроенный Web UI |
| **Pipeline в отдельном thread** с собственным event loop | Изоляция от uvicorn event loop; предотвращение deadlock |
| **Semaphore(1)** для pipeline | GPU-bound: одновременно 1 pipeline; предотвращение OOM |
| **Semaphore(4)** для TTS-сегментов | Параллельный синтез до 4 сегментов внутри 1 pipeline |
| **Redis** для session state (planned) | TTL-based session management; масштабирование в production |
| **In-memory session store** (PoC) | Быстрый старт без внешних зависимостей |

---

## 2. Список модулей и их роли

### 2.1 Модульная карта

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer (M5)                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ FastAPI       │ │ WebSocket    │ │ Web UI (embedded)│ │
│  │ REST          │ │ Streaming    │ │ HTML/JS/CSS      │ │
│  └──────────────┘ └──────────────┘ └──────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Security Layer (M4)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ Input         │ │ PII Masker   │ │ Voice Whitelist  │ │
│  │ Sanitizer     │ │ (regex)      │ │                  │ │
│  └──────────────┘ └──────────────┘ └──────────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Orchestrator (LangGraph, M2)               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  GraphState ──→ Director → Actor → Critic ──┐       │ │
│  │                                      │      ▼       │ │
│  │                               ┌──── route ────┐     │ │
│  │                               │    │    │     │     │ │
│  │                            approved hotfix editor max│ │
│  │                              END  Director Editor HuR│ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Agent Layer (M2+M3)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Director  │ │ Actor    │ │ Critic   │ │ Editor     │ │
│  │ LLM text  │ │ TTS synth│ │ ASR+Judge│ │ Repair     │ │
│  │ → segments│ │ → WAV    │ │ → errors │ │ → new WAV  │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Inference Client Layer (M1)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ VLLMClient│ │ TTSClient│ │ ASRClient│ │ ModelReg.  │ │
│  │ OpenAI API│ │ HTTP     │ │ HTTP     │ │ Lifecycle  │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Audio Utilities (M3)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Alignment │ │ Masking  │ │ Crossfade│ │ Metrics    │ │
│  │ ms→mel    │ │ binary   │ │ eq-power │ │ convergence│ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
├─────────────────────────────────────────────────────────┤
│              Observability (M7)                          │
│  ┌──────────┐ ┌──────────────────────┐ ┌──────────────┐ │
│  │ Prometheus│ │ structlog (JSON/dev) │ │ Tracing      │ │
│  │ /metrics  │ │ + service metadata   │ │ (trace_id)   │ │
│  └──────────┘ └──────────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Сводная таблица модулей

| Модуль | Файлы | Роль | Зависимости |
|--------|-------|------|-------------|
| **Director** | `agents/director.py`, `agents/prompts.py` | Текстовый анализ → сегменты + эмоции | VLLMClient |
| **Actor** | `agents/actor.py` | Сегменты → WAV аудио (параллельный синтез) | TTSClient |
| **Critic** | `agents/critic.py`, `agents/prompts.py` | ASR → Judge → ошибки + WER | ASRClient, VLLMClient |
| **Editor** | `agents/editor.py` | Пересинтез неодобренных сегментов | TTSClient |
| **Orchestrator** | `orchestrator/graph.py`, `orchestrator/state.py` | State machine + conditional routing | LangGraph, все агенты |
| **Schemas** | `agents/schemas.py` | Data contracts (Pydantic) | — |
| **VLLMClient** | `inference/vllm_client.py` | LLM инференс (Qwen3-8B) | AsyncOpenAI |
| **TTSClient** | `inference/tts_client.py` | TTS инференс (CosyVoice3) | httpx |
| **ASRClient** | `inference/asr_client.py` | ASR инференс (WhisperX) | httpx |
| **ModelRegistry** | `inference/model_registry.py` | Lifecycle: init → health → shutdown | все клиенты |
| **Audio Utils** | `audio/*.py` | Alignment, masking, crossfade, convergence | numpy |
| **Security** | `security/*.py` | Sanitization, PII, voice whitelist | regex |
| **API** | `api/app.py`, `api/schemas.py`, `api/sessions.py` | REST + WebSocket + Web UI | FastAPI |
| **Monitoring** | `monitoring/__init__.py` | Counter, Gauge, Histogram → Prometheus | — |
| **Config** | `config.py` | Pydantic Settings из env | pydantic-settings |
| **Logging** | `log.py` | structlog: JSON (prod) / console (dev) | structlog |

---

## 3. Основной Workflow выполнения задачи

### 3.1 Execution Flow

```
POST /synthesize { text, voice_id }
  │
  ├── 1. Input Sanitization     → 10 regex patterns, max_length=5000
  ├── 2. PII Masking             → email, phone, card, passport → [MASK]
  ├── 3. Voice Validation        → whitelist check (3 speakers)
  ├── 4. Session Creation        → UUID, state=queued → processing
  ├── 5. Pipeline Launch         → threading.Thread + own event loop
  │     │
  │     ▼
  │   ┌──────────────────────────────────────────────────┐
  │   │  iteration = 0                                    │
  │   │                                                   │
  │   │  Director(Qwen3-8B)                               │
  │   │    text → chat_json() → DirectorOutput            │
  │   │    → segments[] + emotions + phoneme_hints        │
  │   │    → если iteration>0: _apply_hotfix_hints()      │
  │   │                                                   │
  │   │  Actor(CosyVoice3)                                │
  │   │    ∀ segment[i] where !segment_approved[i]:       │
  │   │      asyncio.gather() + Semaphore(4)              │
  │   │      tts.synthesize(text, voice, instruct) → WAV  │
  │   │    segment_audio[] → concat → audio_bytes         │
  │   │                                                   │
  │   │  Critic(WhisperX + Qwen3-8B)                      │
  │   │    ∀ unapproved segment[i]:                       │
  │   │      Phase 1: ASR → transcript + word_timestamps  │
  │   │      Phase 2: Judge → errors + WER + is_approved  │
  │   │    segment_approved[i] = per-segment verdict       │
  │   │    iteration += 1                                  │
  │   │                                                   │
  │   │  route_after_critic() →                           │
  │   │    ├── approved       → END ✅                    │
  │   │    ├── hotfix         → Director (phoneme hints)  │
  │   │    ├── editor         → Editor → Critic           │
  │   │    └── max_retries    → human_review ⚠️           │
  │   │                                                   │
  │   │  Editor(CosyVoice3) [если route=editor]           │
  │   │    failed_segs = _get_failed_segments()           │
  │   │    ∀ failed segment: tts.synthesize() → new WAV   │
  │   │    _rebuild_combined_audio()                      │
  │   │    convergence_score()                            │
  │   │    → обратно к Critic                             │
  │   └──────────────────────────────────────────────────┘
  │
  ├── 6. Session Update          → status=completed, audio_bytes
  └── 7. Response/WebSocket      → agent_log stream
```

### 3.2 Routing Logic (подробно)

| Условие | Route | Действие |
|---------|-------|----------|
| `is_approved == True` | `approved` | → END, аудио готово |
| `iteration >= max_retries` | `needs_human_review` | → END, флаг эскалации |
| Все ошибки неодобренных сегментов имеют `can_hotfix=True` | `hotfix` | → Director с phoneme hints |
| Существуют ошибки с `can_hotfix=False` | `editor` | → Editor, пересинтез сегментов |

### 3.3 Типичные сценарии

| Сценарий | Итерации | WER | Размер |
|----------|---------|-----|--------|
| Короткий текст (~50 символов), чистый синтез | 1 | 0.000 | ~150 KB |
| Средний текст (~500 символов), 1 ошибка | 2 | 0.000 | ~2 MB |
| Длинный текст (~800 символов), сложные слова | 2-3 | 0.000 | ~2.5 MB |
| Максимально сложный, max_retries | 5 | >0 | — (human review) |

---

## 4. State / Memory / Context Handling

### 4.1 GraphState — центральная структура

`GraphState` (Pydantic `BaseModel`) — единственный объект передаваемый между узлами LangGraph. Содержит:

| Группа | Поля | Описание |
|--------|------|----------|
| **Input** | `text`, `voice_id`, `trace_id` | Входные данные пользователя |
| **Director** | `ssml_markup`, `tts_instruct` | Структурированные инструкции синтеза |
| **Actor** | `audio_bytes`, `sample_rate`, `segment_audio[]`, `segment_approved[]` | Аудио-данные |
| **Critic** | `transcript`, `word_timestamps[]`, `errors[]`, `wer`, `is_approved` | Результаты оценки |
| **Control** | `iteration`, `max_retries`, `needs_human_review`, `convergence_score` | Управление потоком |
| **Log** | `agent_log[]` | Журнал действий агентов |

### 4.2 Memory policy

| Аспект | Реализация | Ограничение |
|--------|-----------|-------------|
| **Session memory** | In-memory `SessionStore` (dict) | Теряется при перезапуске; PoC only |
| **Cross-session** | Нет | Каждый запрос обрабатывается с нуля |
| **Agent memory** | Нет persistent memory | Агенты stateless между запросами |
| **Ephemeral data** | WAV, промежуточные результаты — в RAM | Удаляется после завершения сессии |
| **Логирование** | Только анонимизированные метаданные | PII не пишутся в логи (planned) |

### 4.3 Context budget

| Контекст | Бюджет | Обоснование |
|----------|--------|-------------|
| vLLM `max-model-len` | **16384 tokens** | Достаточно для длинных промптов Director + Judge |
| Director prompt + text | ~2000-3000 tokens | System prompt + user text до 5000 chars |
| Judge prompt + target + transcript | ~1000-2000 tokens | Обрезка до 500 chars + max 10 timestamps |
| `max_tokens` response | **4096 tokens** | Максимум для JSON ответа |

---

## 5. Retrieval-контур

### 5.1 Текущее состояние

В текущей PoC-системе **retrieval-контура нет**. Система работает в generative-only режиме:

- Director генерирует сегментацию на основе промпта и входного текста
- Critic оценивает quality через ASR + LLM-Judge
- Нет внешних knowledge bases, embedding stores или vector databases

### 5.2 Retrieval-like механизмы

| Механизм | Описание | Статус |
|----------|----------|--------|
| **Phoneme hint lookup** | Director может использовать `phoneme_hints` из предыдущих ошибок Critic | ✅ Реализовано (intra-session) |
| **Segment cache** | Actor переиспользует `segment_audio[i]` если `segment_approved[i]=True` | ✅ Реализовано (intra-session) |
| **Voice lookup** | `VOICE_MAP` в TTSClient для маппинга `voice_id → speaker name` | ✅ Реализовано |
| **Pronunciation memory** | Cross-session кэш (word, voice) → phoneme_hint | 🟡 Planned (MAS-4) |
| **Segment embedding cache** | hash(text + voice + emotion) → audio + WER | 🟡 Planned (MAS-4) |

### 5.3 Планируемый retrieval-контур (MAS-4)

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Director    │────▶│ Pronunciation    │────▶│ Redis/SQLite │
│              │     │ Memory Retrieval │     │ Key-Value    │
│ "Какой hint  │     │ (word, voice_id) │     │ Store        │
│  работал    │     │  → phoneme_hint  │     │              │
│  раньше?"   │     │  → success_rate  │     │              │
└──────────────┘     └──────────────────┘     └──────────────┘
                                                     │
                     ┌──────────────────┐            │
                     │ Segment Cache    │◀───────────┘
                     │ hash(text+voice+ │
                     │ emotion) → {WAV, │
                     │ WER, timestamp}  │
                     └──────────────────┘
```

---

## 6. Tool/API-интеграции

### 6.1 Внутренние модельные API

| Service | Protocol | Endpoint | Used by | Timeout | Retry |
|---------|----------|----------|---------|---------|-------|
| **vLLM** (Qwen3-8B) | OpenAI-compat HTTP | `:8055/v1/chat/completions` | Director, Critic | 300s | 5× exponential |
| **CosyVoice3** (0.5B) | HTTP/REST | `:9880/tts` | Actor, Editor | 60s | — |
| **WhisperX** (large-v3) | HTTP/REST | `:9881/transcribe` | Critic | 60s | — |

### 6.2 Внешние API / Side effects

| API | Назначение | Side effects | Защита |
|-----|-----------|-------------|--------|
| **Redis** (`:8056`) | Session storage (planned) | Запись/чтение TTL-based sessions | maxmemory=256MB, allkeys-lru |
| Нет облачных API | Весь inference локальный | — | — |

### 6.3 Контракты (подробнее в `docs/specs/tools-apis.md`)

**VLLMClient:**
- `chat(system_prompt, user_message) → str`
- `chat_json(system_prompt, user_message, response_model) → T`
- Fallback: `<think>` stripping → JSON parse → brace extraction → error

**TTSClient:**
- `synthesize(text, voice_id, instruct?) → AudioResult(waveform, sample_rate)`
- `clone_voice(text, ref_audio, ref_text) → AudioResult`
- Guard: voice whitelist validation перед вызовом

**ASRClient:**
- `transcribe(audio_bytes, sample_rate) → TranscriptionResult(text, word_timestamps[], language)`

---

## 7. Failure Modes, Fallback и Guardrails

### 7.1 Failure modes

| Failure | Вероятность | Impact | Detection | Mitigation |
|---------|------------|--------|-----------|------------|
| **Infinite correction loop** | ⚠️ Средняя | OOM, GPU waste | `iteration >= max_retries` | Hard limit `MAX_RETRIES=5`, escalation to human review |
| **vLLM connection timeout** | ⚠️ Средняя | Pipeline stall | `APIConnectionError` / `APITimeoutError` | 5× retry с exponential backoff (2^n sec); 300s hard timeout |
| **JSON parse failure** (Qwen3 `<think>`) | ⚠️ Средняя | Pipeline crash | `json.JSONDecodeError` | 3-step fallback: strip `<think>` → parse → brace extraction |
| **CosyVoice assertion** (instruct token) | Низкая | TTS failure | `AssertionError` | Auto-append `<\|endofprompt\|>` в server.py |
| **Pipeline timeout** (300s) | Низкая | Session stuck | `asyncio.wait_for` timeout | Graceful failure → `status=failed`, pipeline_semaphore release |
| **Prompt injection** | ⚠️ Средняя | Unsafe LLM behavior | 10 compiled regex patterns | `sanitize_input()` → reject с 400 |
| **PII leakage** | ⚠️ Средняя | Privacy violation | Regex: email/phone/card/passport/INN/IP | `mask_pii()` → `[EMAIL_1]`, `[PHONE_1]` |
| **Voice spoofing** | Низкая (PoC) | Deepfake risk | Voice whitelist check | Only 3 predefined voices; zero-shot cloning disabled |
| **GPU OOM** | Низкая | Container crash | Docker healthcheck | Semaphore(1) — 1 pipeline at a time; GPU memory utilization=0.7 |
| **Event loop deadlock** | Низкая | App hangs | Health check fails | Pipeline в отдельном Thread с собственным event loop |

### 7.2 Guardrails

| Guardrail | Реализация | Конфигурация |
|-----------|-----------|-------------|
| **Text length** | `max_length=5000` в sanitizer | `SECURITY_MAX_TEXT_LENGTH` |
| **Retry limit** | `max_retries=5` | `SECURITY_MAX_RETRIES` |
| **Pipeline concurrency** | `threading.Semaphore(1)` | Hardcoded |
| **TTS concurrency** | `asyncio.Semaphore(4)` | `max_concurrency` param |
| **Pipeline timeout** | `asyncio.wait_for(300s)` | Hardcoded |
| **Voice whitelist** | 3 speakers | `SECURITY_WHITELISTED_VOICES` |
| **Unknown emotion → neutral** | Validator в `Segment` | `_fallback_unknown_emotion()` |
| **Empty JSON fallback** | Brace extraction в VLLMClient | Automatic |
| **WER threshold for escalation** | 0.15 | `SECURITY_WER_THRESHOLD_FOR_HUMAN_REVIEW` |
| **Judge input truncation** | 500 chars + 10 timestamps | In prompt |

### 7.3 Escalation path

```
Critic not approved
  └── iteration < max_retries?
        ├── YES: hotfix/editor loop continues
        └── NO: mark_human_review()
                  → needs_human_review = True
                  → agent_log: "escalated"
                  → status = completed (с предупреждением)
                  → UI: "Approve Audio with Errors" / "Edit Original Text"
```

---

## 8. Технические и операционные ограничения

### 8.1 Latency

| Операция | p50 | p95 | Bottleneck |
|----------|-----|-----|-----------|
| Director (Qwen3-8B, JSON) | 2-5s | 8-15s | GPU inference, token generation |
| Actor (CosyVoice3, 1 сегмент) | 1-3s | 5-8s | TTS inference, audio encoding |
| Actor (CosyVoice3, 6 сегментов, parallel) | 3-6s | 10-15s | GPU parallelism (Semaphore 4) |
| Critic Phase 1 (WhisperX ASR) | 1-2s | 3-5s | ASR inference |
| Critic Phase 2 (Judge, JSON) | 2-4s | 6-10s | GPU inference |
| Editor (1 segment re-synth) | 1-3s | 5-8s | TTS inference |
| **Full pipeline (1 iteration, short text)** | **8-15s** | **20-30s** | — |
| **Full pipeline (3 iterations, long text)** | **30-60s** | **90-120s** | — |
| **Hard timeout** | — | **300s** | — |

> **RTF (Real-Time Factor)** для 10s аудио: p50 ≈ 1.5-3.0, p95 ≈ 3.0-6.0

### 8.2 Cost (PoC: self-hosted GPU)

| Ресурс | Потребление | Стоимость |
|--------|------------|-----------|
| GPU 1 (A4000 16GB) | vLLM: ~5 GB VRAM, utilization 70% | Амортизация оборудования |
| GPU 2 (A4000 16GB) | CosyVoice3 (~2 GB) + WhisperX (~3 GB) | Амортизация оборудования |
| CPU | FastAPI + orchestrator | Минимальная |
| RAM | Session store + intermediate audio | ~1-2 GB per session peak |
| Storage | Redis 256MB + session WAV | Ephemeral, TTL 1h |

**Стоимость одного запроса (inference tokens):**
- Director: ~1000-2000 output tokens
- Judge: ~500-1000 output tokens per segment
- При 3 итерациях × 6 сегментов: ~10K-20K tokens total
- Self-hosted: $0 marginal (при наличии GPU)

### 8.3 Reliability

| Метрика | Текущее | Целевое |
|---------|---------|---------|
| Pipeline success rate | ≈ 95% (PoC) | > 99% |
| WER на одобренных | 0.000 (observed) | < 0.01 |
| Human acceptance rate | — (нет данных) | > 95% |
| Avg iterations to converge | 1-3 | < 2.5 |
| Uptime | — | 99.9% (with docker restart) |

### 8.4 Limits & Quotas

| Параметр | Значение | Настройка |
|----------|---------|-----------|
| Max text length | 5000 chars | `SECURITY_MAX_TEXT_LENGTH` |
| Max retries | 5 | `SECURITY_MAX_RETRIES` |
| Max concurrent pipelines | 1 | `Semaphore(1)` |
| Max concurrent TTS segments | 4 | `Semaphore(4)` |
| Pipeline hard timeout | 300s | Hardcoded |
| Session TTL | 1h | `REDIS_SESSION_TTL_SECONDS` |
| API rate limit | 10 req/min | `API_RATE_LIMIT_PER_MINUTE` |
| Max JSON response tokens | 4096 | `VLLM_MAX_TOKENS` |
| vLLM context window | 16384 tokens | `max-model-len` |

---

## 9. Ссылки на детальные спецификации

| Спецификация | Файл |
|-------------|------|
| Retriever / Retrieval | [docs/specs/retriever.md](specs/retriever.md) |
| Tools / APIs | [docs/specs/tools-apis.md](specs/tools-apis.md) |
| Memory / Context | [docs/specs/memory-context.md](specs/memory-context.md) |
| Agent / Orchestrator | [docs/specs/agent-orchestrator.md](specs/agent-orchestrator.md) |
| Serving / Config | [docs/specs/serving-config.md](specs/serving-config.md) |
| Observability / Evals | [docs/specs/observability-evals.md](specs/observability-evals.md) |
| Диаграммы (C4 + Flow) | [docs/diagrams/](diagrams/) |
