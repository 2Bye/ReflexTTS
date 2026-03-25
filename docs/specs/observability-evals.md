# Spec: Observability / Evals

> Метрики, логи, трейсы и проверки, которые собираются.

---

## 1. Prometheus Metrics

**Файл:** `src/monitoring/__init__.py`
**Endpoint:** `GET /metrics`
**Format:** Prometheus text exposition

### Метрики

| Metric | Type | Labels | Описание |
|--------|------|--------|----------|
| `reflex_requests_total` | Counter | `voice_id` | Общее число запросов |
| `reflex_request_latency_seconds` | Histogram | — | E2E latency per request |
| `reflex_pipeline_iterations` | Histogram | — | Число итераций до завершения |
| `reflex_pipeline_wer` | Histogram | — | Распределение WER при завершении |
| `reflex_pipeline_status_total` | Counter | `status` | completed / failed / escalated |
| `reflex_active_sessions` | Gauge | — | Число активных сессий (0 или 1 в PoC) |
| `reflex_agent_latency_seconds` | Histogram | — | Latency per agent step |
| `reflex_errors_total` | Counter | `type` | Число ошибок по типам |

### Histogram Buckets

| Метрика | Buckets |
|---------|---------|
| `request_latency` | 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0 |
| `pipeline_iterations` | 1.0, 2.0, 3.0, 4.0, 5.0 |
| `pipeline_wer` | 0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0 |
| `agent_latency` | 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0 |

### Пример вывода `/metrics`

```
reflex_requests_total{voice_id="speaker_1"} 42

reflex_request_latency_seconds_bucket{le="1.0"} 5
reflex_request_latency_seconds_bucket{le="10.0"} 30
reflex_request_latency_seconds_bucket{le="+Inf"} 42
reflex_request_latency_seconds_sum 285.3
reflex_request_latency_seconds_count 42

reflex_pipeline_status_total{status="completed"} 38
reflex_pipeline_status_total{status="failed"} 4

reflex_active_sessions 0
```

---

## 2. Structured Logging

**Файл:** `src/log.py`
**Library:** structlog
**Formats:** JSON (production), colored console (dev)

### Log Events

| Event | Level | Agent | Описание |
|-------|-------|-------|----------|
| `director_start` | INFO | Director | Начало анализа текста |
| `director_llm_response` | INFO | Director | LLM ответ (segments count) |
| `director_segment` | INFO | Director | Детали сегмента |
| `director_done` | INFO | Director | Завершение с instruct |
| `hotfix_applied` | DEBUG | Director | Phoneme hint injected |
| `vllm_request_success` | DEBUG | VLLMClient | Attempt, tokens used |
| `vllm_request_retry` | WARNING | VLLMClient | Retry attempt |
| `vllm_json_parse_retry` | WARNING | VLLMClient | JSON fallback |
| `vllm_json_parse_error` | ERROR | VLLMClient | Finall parse fail |
| `route_approved` | INFO | Orchestrator | Audio approved |
| `route_editor` | INFO | Orchestrator | Editor routing |
| `route_hotfix` | INFO | Orchestrator | Hotfix routing |
| `route_max_retries` | WARNING | Orchestrator | Max retries reached |
| `pipeline_escalated` | WARNING | Orchestrator | Human review needed |
| `pipeline_start` | INFO | API | Pipeline started |
| `pipeline_completed` | INFO | API | Pipeline done |
| `pipeline_failed` | ERROR | API | Pipeline error |
| `pipeline_timeout` | ERROR | API | 300s timeout |
| `injection_detected` | WARNING | Security | Prompt injection found |
| `input_too_long` | WARNING | Security | Text exceeds limit |
| `app_created` | INFO | API | Server initialized |

### Log Format (JSON production)

```json
{
  "event": "pipeline_completed",
  "logger": "src.api.app",
  "level": "info",
  "timestamp": "2026-03-25T10:30:00Z",
  "service": "reflex-tts",
  "session_id": "a3f1b2c4-...",
  "wer": 0.0,
  "is_approved": true,
  "iterations": 2,
  "audio_size_kb": 150
}
```

### Log Metadata

| Поле | Источник | Описание |
|------|---------|----------|
| `service` | `LOG_SERVICE_NAME` | Константа "reflex-tts" |
| `logger` | `__name__` | Python module path |
| `level` | structlog | info/warning/error/debug |
| `timestamp` | structlog ISO | UTC timestamp |

---

## 3. Tracing

### Текущее состояние

| Аспект | Реализация |
|--------|-----------|
| **trace_id** | `GraphState.trace_id` = session UUID | 
| **Propagation** | Передаётся через GraphState |
| **OpenTelemetry** | Конфигурация есть (`LOG_ENABLE_OTEL`), интеграция **не реализована** |
| **Distributed tracing** | Нет |

### Планируемое (production)

```
FastAPI → OpenTelemetry → Jaeger/Tempo
  │
  ├── Span: POST /synthesize  (session_id)
  │   ├── Span: director_node  (duration, segments)
  │   ├── Span: actor_node     (duration, segments_count)
  │   ├── Span: critic_node    (duration, wer, approved)
  │   └── Span: editor_node    (duration, failed_segments)
  │
  ├── Span: vLLM chat_json    (tokens, model, attempt)
  ├── Span: CosyVoice synth   (text_len, duration_s)
  └── Span: WhisperX transcribe (audio_duration, confidence)
```

---

## 4. Evaluations / Quality Checks

### Automated Quality Metrics

| Метрика | Формула | Порог | Описание |
|---------|---------|-------|----------|
| **WER** | Levenshtein(target, transcript) / len(target) | < 0.01 target | Word Error Rate |
| **Convergence** | `0.5(1-WER) + 0.3×SECS + 0.2×(PESQ/4.5)` | ≥ 0.85 | Composite quality |
| **SECS** | cosine_similarity(speaker_emb, ref_emb) | > 0.85 | Voice consistency |
| **PESQ** | ITU-T P.862 | > 3.0 / 4.5 | Perceptual quality |

### Benchmark Suite

**Файлы:** `scripts/benchmark_texts.json`, `scripts/run_benchmarks.py`

| Параметр | Значение |
|----------|---------|
| Количество текстов | 47 (EN) |
| Категории | short / medium / long |
| Метрики в отчёте | WER, latency, RTF, iterations |
| Нагрузочное тестирование | `scripts/load_test.py` (Locust) |

### Test Suite

| Категория | Файлы | Тесты |
|----------|-------|-------|
| Config | `test_config.py` | 8 |
| Logging | `test_logging.py` | 4 |
| Inference | `test_vllm_client.py`, `test_tts_client.py`, `test_asr_client.py`, `test_model_registry.py` | 24 |
| Agents | `test_schemas.py`, `test_director.py`, `test_actor.py`, `test_critic.py`, `test_editor.py` | 27 |
| Graph | `test_graph.py` | 4 |
| Audio | `test_audio.py` | 18 |
| Security | `test_security.py` | 25 |
| API | `test_api.py` | 17 |
| Benchmarks | `test_benchmarks.py` | 12 |
| Monitoring | `test_monitoring.py` | 10 |
| **Total** | — | **171** |

### CI/CD Pipeline

```
.github/workflows/ci.yml:
  1. ruff check (lint)
  2. mypy --strict (type check)
  3. bandit (security scan)  
  4. detect-secrets (secret detection)
  5. pytest (unit tests, 171)
```

---

## 5. Alerting (planned)

| Alert | Condition | Severity |
|-------|-----------|----------|
| Pipeline success rate < 90% | `reflex_pipeline_status{status="failed"}` / total > 10% | Critical |
| Pipeline latency p95 > 120s | `reflex_request_latency_seconds` p95 > 120 | Warning |
| Active sessions stuck | `reflex_active_sessions` > 0 for > 10min | Critical |
| WER regression | `reflex_pipeline_wer` p50 > 0.05 | Warning |
| vLLM health fails | Docker healthcheck | Critical |
| GPU OOM | Container restart + memory metrics | Critical |

---

## 6. Dashboard (planned)

```
┌────────────────────────────────────────────────────────┐
│                ReflexTTS Dashboard                      │
├────────────────────────┬───────────────────────────────┤
│  Active Sessions:  0   │  Total Requests:   1,247      │
│  Success Rate:   96.8% │  Avg Latency:     18.3s       │
│  Avg WER:      0.002   │  Avg Iterations:    1.7       │
├────────────────────────┴───────────────────────────────┤
│  [Graph: Request Latency over time (p50/p95/p99)]      │
│  [Graph: WER distribution histogram]                   │
│  [Graph: Pipeline iterations histogram]                │
│  [Graph: Status breakdown (completed/failed/escalated)]│
│  [Graph: Agent latency breakdown (dir/act/crit/edit)]  │
└────────────────────────────────────────────────────────┘
```
