# ReflexTTS — Обзор проекта

> Самокорректирующийся TTS с агентным пайплайном: Director → Actor → Critic → Editor.
> Стек: **Qwen3-8B** (vLLM) + **CosyVoice3** (0.5B) + **WhisperX** — всё локально.

---

## Текущий статус

| Milestone | Статус | Тесты | Описание |
|-----------|--------|-------|----------|
| M0: Инфраструктура | ✅ | 12 | Структура, config, Docker, CI/CD |
| M1: Inference Backend | ✅ | 45 | vLLM, CosyVoice3, WhisperX клиенты |
| M2: Агентный пайплайн | ✅ | 73 | Director, Actor, Critic, LangGraph |
| M3: Editor + Inpainting | ✅ | 94 | Editor, audio utils, convergence |
| M4: Security & Governance | ✅ | 119 | Sanitizer, PII masker, whitelist |
| M5: API & Web UI | ✅ | 136 | FastAPI, WebSocket, Web UI |
| M6: Benchmarking | ✅ | 148 | 50 texts, benchmark runner, locust |
| M7: Production | ✅ | 158 | Prometheus metrics, /metrics |
| M8: E2E Integration | ✅ | 158 | Full pipeline: vLLM + CosyVoice3 + WhisperX |

**Проверки:** `ruff check` ✅ · `mypy --strict` ✅ (33 файла) · `pytest` ✅ (158/158)

---

## Архитектура

```mermaid
graph TD
    UI["Web UI / API<br/>(M5)"] --> Graph["LangGraph Orchestrator"]

    Graph --> Dir["Director Agent<br/>text → segments + emotions"]
    Graph --> Act["Actor Agent<br/>segments → WAV audio"]
    Graph --> Crit["Critic Agent<br/>audio → transcript → errors"]
    Graph --> Ed["Editor Agent<br/>errors → repaired audio"]

    Dir -.-> LLM["Qwen3-8B<br/>(vLLM)"]
    Crit -.->|Judge| LLM
    Act -.-> TTS["CosyVoice3<br/>0.5B"]
    Ed -.-> TTS
    Crit -.->|ASR| WX["WhisperX"]

    Crit -->|approved| Done["✅ Final Audio"]
    Crit -->|hotfix| Dir
    Crit -->|editor| Ed
    Ed --> Crit
    Crit -->|max retries| Human["⚠️ Human Review"]
```

---

## Карта файлов

### `src/` — исходный код

```
src/
├── config.py                    # Pydantic Settings — все конфиги из env
├── log.py                       # structlog: JSON (prod) / console (dev)
│
├── inference/                   # M1 — клиенты к моделям
│   ├── __init__.py              # Публичный API пакета
│   ├── vllm_client.py           # Async OpenAI-compatible → Qwen3-8B
│   ├── tts_client.py            # CosyVoice3 AutoModel wrapper
│   ├── asr_client.py            # WhisperX + forced alignment
│   └── model_registry.py        # Lifecycle: init → health → shutdown
│
├── agents/                      # M2 + M3 — агенты
│   ├── __init__.py              # Экспорт типов
│   ├── schemas.py               # DirectorOutput, CriticOutput, Segment
│   ├── prompts.py               # System prompts для Qwen3-8B (JSON)
│   ├── director.py              # Текст → сегменты + эмоции + phoneme hints
│   ├── actor.py                 # Сегменты → WAV (concat + pause + encode)
│   ├── critic.py                # ASR → Judge → ошибки + WER
│   └── editor.py                # Inpainting (FM) / chunk regen (fallback)
│
├── audio/                       # M3 — аудио-утилиты
│   ├── __init__.py
│   ├── alignment.py             # ms → mel-frame indices, MelRegion
│   ├── masking.py               # Binary mask + cosine taper
│   ├── crossfade.py             # Equal-power cross-fade
│   └── metrics.py               # Convergence score (WER+SECS+PESQ)
│
├── orchestrator/                # M2 — LangGraph
│   ├── __init__.py
│   ├── state.py                 # GraphState, DetectedError, AgentLogEntry
│   └── graph.py                 # build_graph() — 4-way routing
│
├── api/                         # M5 — API + Web UI
│   ├── __init__.py
│   ├── app.py                   # FastAPI + all endpoints + embedded UI
│   ├── schemas.py               # SynthesizeRequest, SessionStatus
│   └── sessions.py              # In-memory session store (PoC)
│
├── security/                    # M4 — Security
│   ├── __init__.py
│   ├── input_sanitizer.py       # Prompt injection guard (10 patterns)
│   ├── pii_masker.py            # Email/phone/card/passport/INN/IP mask
│   └── voice_whitelist.py       # Whitelist + clone blocking
│
└── monitoring/                  # M7 — Prometheus metrics
    └── __init__.py              # Counter, Gauge, Histogram, MetricsRegistry
```

### `tests/` — тесты

```
tests/
└── unit/
    ├── test_config.py           # 8 тестов — конфигурация
    ├── test_logging.py          # 4 теста — логирование
    ├── test_vllm_client.py      # 8 тестов — chat, JSON, retries, health
    ├── test_tts_client.py       # 8 тестов — AudioResult, voices, guards
    ├── test_asr_client.py       # 4 теста — WordTimestamp, guards
    ├── test_model_registry.py   # 4 теста — lifecycle
    ├── test_schemas.py          # 10 тестов — Pydantic validation
    ├── test_director.py         # 3 теста — segments, hotfix injection
    ├── test_actor.py            # 6 тестов — WAV encoding/decoding
    ├── test_critic.py           # 5 тестов — WAV decode, text extraction
    ├── test_graph.py            # 4 теста — graph construction, state
    ├── test_audio.py            # 18 тестов — alignment, mask, crossfade, metrics
    ├── test_editor.py           # 3 теста — skip paths, fallback
    ├── test_security.py         # 25 тестов — injection, PII, voice whitelist
    ├── test_api.py              # 17 тестов — endpoints, sessions, schemas
    ├── test_benchmarks.py       # 12 тестов — texts, summary, report
    └── test_monitoring.py       # 10 тестов — Counter, Gauge, Histogram
```

### `docker/` + CI/CD

```
docker/
├── docker-compose.yml           # vLLM + Redis + App
└── Dockerfile.app               # Multi-stage, non-root

.github/workflows/ci.yml         # lint → type-check → security → tests
.pre-commit-config.yaml          # Ruff, Mypy, Bandit, Detect Secrets
.env.example                     # Шаблон переменных окружения
```

### `docs/` — документация

```
docs/
├── governance.md                # Governance-модель
├── product-proposal.md          # Продуктовое описание
├── latent-inpainting-architecture.md  # Архитектура inpainting
├── walkthrough-m0-infrastructure.md   # Walkthrough M0
├── walkthrough-m1-inference.md        # Walkthrough M1
├── walkthrough-m2-agents.md           # Walkthrough M2
├── walkthrough-m3-editor.md           # Walkthrough M3
├── walkthrough-m4-security.md         # Walkthrough M4
├── walkthrough-m5-api.md              # Walkthrough M5
├── walkthrough-m6-benchmarks.md       # Walkthrough M6
└── walkthrough-m7-production.md       # Walkthrough M7
```

### `scripts/` — M6 benchmarks

```
scripts/
├── benchmark_texts.json     # 50 тестовых текстов (RU/EN/ZH/mixed)
├── run_benchmarks.py        # Прогон + WER/latency/RTF отчёт
└── load_test.py             # Locust нагрузочный тест
```

---

## Потоки данных в пайплайне

```
User text
  │
  ▼
┌─────────────────────────────────────────────┐
│ Director (Qwen3-8B)                         │
│ Input:  text                                │
│ Output: DirectorOutput (segments, emotions) │
│         → ssml_markup, tts_instruct         │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Actor (CosyVoice3)                          │
│ Input:  ssml_markup + tts_instruct          │
│ Output: audio_bytes (WAV 16-bit PCM)        │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Critic (WhisperX → Qwen3-8B Judge)          │
│ Phase 1: ASR → transcript + word_timestamps │
│ Phase 2: Judge → errors + wer + is_approved │
└──────────────────┬──────────────────────────┘
                   ▼
          ┌────────┼────────┐
          ▼        ▼        ▼
       Approved  Hotfix   Editor
       (END)     (→Dir)   (→Critic)
```

### GraphState — ключевые поля

| Поле | Тип | Кто пишет | Кто читает |
|------|-----|----------|-----------|
| `text` | str | User | Director |
| `voice_id` | str | User | Actor |
| `ssml_markup` | dict | Director | Actor |
| `tts_instruct` | str | Director | Actor |
| `audio_bytes` | bytes | Actor/Editor | Critic |
| `transcript` | str | Critic (ASR) | Critic (Judge) |
| `word_timestamps` | list | Critic (ASR) | Critic (Judge) |
| `errors` | list[DetectedError] | Critic (Judge) | Editor/Director |
| `wer` | float | Critic | Graph routing |
| `is_approved` | bool | Critic | Graph routing |
| `iteration` | int | Graph | Graph routing |
| `convergence_score` | float | Editor | Observability |

---

## GPU Budget (~10 GB)

| Модель | Компонент | VRAM |
|--------|----------|------|
| Qwen3-8B-Instruct AWQ 4-bit | Director + Critic Judge | ~5 GB |
| CosyVoice3 0.5B | Actor + Editor | ~2 GB |
| WhisperX (large-v3 + Wav2Vec2) | Critic ASR | ~3 GB |

---

## Как запустить

```bash
# Установка зависимостей
pip install -e ".[dev]"

# Проверки
ruff check src/ tests/         # Lint
python -m mypy src/             # Type check
python -m pytest tests/unit/ -v # Тесты (158 шт)

# Запуск Web UI (без GPU)
python -m uvicorn src.api.app:create_app --factory --port 8080
# Откройте http://localhost:8080

# Prometheus метрики
curl http://localhost:8080/metrics

# Docker (с GPU)
cd docker && docker compose up -d

# Бенчмарки (когда стек поднят)
python scripts/run_benchmarks.py
```

## E2E интеграция (M8)

Полный пайплайн протестирован end-to-end с реальными GPU-сервисами:

| Сервис | Порт | GPU | Модель |
|--------|------|-----|--------|
| vLLM | :8055 | GPU 1 | Qwen3-8B-AWQ (16k ctx) |
| CosyVoice3 | :9880 | GPU 2 | Fun-CosyVoice3-0.5B |
| WhisperX | :9881 | GPU 2 | large-v3 |
| App (FastAPI) | :8081 | CPU | — |
| Redis | :8056 | CPU | — |

**Результаты:**
- Короткий текст: Director → Actor → Critic → ✅ approved (WER=0.0, ~15s)
- Длинный текст: Director (Qwen3) выделяет сегменты → Actor (CosyVoice3) синтезирует → Critic (WhisperX + Qwen3 Judge) верифицирует

```bash
# Запуск всего стека
cd docker && docker compose up -d

# E2E тест
curl -X POST http://localhost:8081/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world.", "voice_id": "speaker_1"}'

# Логи агентов
docker logs reflex-app -f
```
