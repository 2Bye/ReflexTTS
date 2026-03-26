# Spec: Observability / Evals

> Metrics, logs, traces, and quality checks.

---

## 1. Prometheus Metrics

**File:** `src/monitoring/__init__.py`
**Endpoint:** `GET /metrics`
**Format:** Prometheus text exposition

### Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `reflex_requests_total` | Counter | `voice_id` | Total request count |
| `reflex_request_latency_seconds` | Histogram | — | E2E latency per request |
| `reflex_pipeline_iterations` | Histogram | — | Iterations until completion |
| `reflex_pipeline_wer` | Histogram | — | WER distribution at completion |
| `reflex_pipeline_status_total` | Counter | `status` | completed / failed / escalated |
| `reflex_active_sessions` | Gauge | — | Active sessions count (0 or 1 in PoC) |
| `reflex_agent_latency_seconds` | Histogram | — | Latency per agent step |
| `reflex_errors_total` | Counter | `type` | Error count by type |

### Histogram Buckets

| Metric | Buckets |
|--------|---------|
| `request_latency` | 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0 |
| `pipeline_iterations` | 1.0, 2.0, 3.0, 4.0, 5.0 |
| `pipeline_wer` | 0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0 |
| `agent_latency` | 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0 |

### Example `/metrics` Output

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

**File:** `src/log.py`
**Library:** structlog
**Formats:** JSON (production), colored console (dev)

### Log Events

| Event | Level | Agent | Description |
|-------|-------|-------|-------------|
| `director_start` | INFO | Director | Text analysis started |
| `director_llm_response` | INFO | Director | LLM response (segments count) |
| `director_segment` | INFO | Director | Segment details |
| `director_done` | INFO | Director | Completion with instruct |
| `hotfix_applied` | DEBUG | Director | Phoneme hint injected |
| `vllm_request_success` | DEBUG | VLLMClient | Attempt, tokens used |
| `vllm_request_retry` | WARNING | VLLMClient | Retry attempt |
| `vllm_json_parse_retry` | WARNING | VLLMClient | JSON fallback |
| `vllm_json_parse_error` | ERROR | VLLMClient | Final parse fail |
| `route_approved` | INFO | Orchestrator | Audio approved |
| `route_editor` | INFO | Orchestrator | Editor routing |
| `route_hotfix` | INFO | Orchestrator | Hotfix routing |
| `route_max_retries` | WARNING | Orchestrator | Max retries reached |
| `pipeline_escalated` | WARNING | Orchestrator | Human review needed |
| `pipeline_start` | INFO | API | Pipeline started |
| `pipeline_dequeued` | INFO | API | Session dequeued from pipeline |
| `pipeline_completed` | INFO | API | Pipeline done |
| `pipeline_failed` | ERROR | API | Pipeline error |
| `pipeline_timeout` | ERROR | API | 300s timeout |
| `rate_limit_exceeded` | WARNING | API | IP rate limit exceeded |
| `pronunciation_cache_hit` | DEBUG | Director | Cached hint applied |
| `pronunciation_cache_record` | INFO | Critic | Hint result recorded |
| `segment_cache_hit` | DEBUG | Actor | Cached audio used |
| `segment_cache_store` | INFO | Actor | Audio cached |
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

| Field | Source | Description |
|-------|--------|-------------|
| `service` | `LOG_SERVICE_NAME` | Constant "reflex-tts" |
| `logger` | `__name__` | Python module path |
| `level` | structlog | info/warning/error/debug |
| `timestamp` | structlog ISO | UTC timestamp |

---

## 3. Tracing

### Current State

| Aspect | Implementation |
|--------|---------------|
| **trace_id** | `GraphState.trace_id` = session UUID |
| **Propagation** | Passed through GraphState |
| **OpenTelemetry** | ✅ Implemented (`src/monitoring/tracing.py`). `LOG_ENABLE_OTEL=true` → TracerProvider + OTLP/Console exporter |
| **Distributed tracing** | ✅ OTel spans for agent nodes (via `get_tracer()`) |

### Span Hierarchy (production)

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

> ✅ **Implemented**: `src/monitoring/tracing.py` — `init_tracing(config)` + `get_tracer(name)`. NoOp when `LOG_ENABLE_OTEL=false`.

---

## 4. Evaluations / Quality Checks

### Automated Quality Metrics

| Metric | Formula | Threshold | Description |
|--------|---------|-----------|-------------|
| **WER** | Levenshtein(target, transcript) / len(target) | < 0.01 target | Word Error Rate |
| **Convergence** | `0.5(1-WER) + 0.3×SECS + 0.2×(PESQ/4.5)` | ≥ 0.85 | Composite quality |
| **SECS** | cosine_similarity(speaker_emb, ref_emb) | > 0.85 | Voice consistency |
| **PESQ** | ITU-T P.862 | > 3.0 / 4.5 | Perceptual quality |

### Benchmark Suite

**Files:** `scripts/benchmark_texts.json`, `scripts/run_benchmarks.py`

| Parameter | Value |
|-----------|-------|
| Number of texts | 47 (EN) |
| Categories | short / medium / long |
| Report metrics | WER, latency, RTF, iterations |
| Load testing | `scripts/load_test.py` (Locust) |

### Test Suite

| Category | Files | Tests |
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
| Rate Limiter | `test_rate_limiter.py` | 5 |
| Pronunciation Cache | `test_pronunciation_cache.py` | 8 |
| Segment Cache | `test_segment_cache.py` | 7 |
| Tracing | `test_tracing.py` | 2 |
| **Total** | — | **193** |

### CI/CD Pipeline

```
.github/workflows/ci.yml:
  1. ruff check (lint)
  2. mypy --strict (type check)
  3. bandit (security scan)  
  4. detect-secrets (secret detection)
  5. pytest (unit tests, 193)
```

---

## 5. Alerting

> ✅ **Implemented**: `docker/prometheus/alerts.yml` + `docker/prometheus/prometheus.yml`

| Alert | Condition | Severity |
|-------|-----------|----------|
| Pipeline success rate < 90% | `rate(reflex_pipeline_status{status="failed"}[5m]) / rate(total) > 0.1` | Critical |
| Pipeline latency p95 > 120s | `histogram_quantile(0.95, reflex_request_latency_seconds_bucket) > 120` | Warning |
| Active sessions stuck | `reflex_active_sessions > 0` for > 10min | Critical |
| WER regression | `reflex_pipeline_wer` p50 > 0.05 | Warning |
| High iteration count | `reflex_pipeline_iterations` p50 > 2.5 | Warning |

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
