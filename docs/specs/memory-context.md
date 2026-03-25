# Spec: Memory / Context

> Session state, memory policy, context budget.

---

## 1. Session State

### SessionStore (PoC — In-Memory)

**Файл:** `src/api/sessions.py`

```python
class SessionState(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Session:
    session_id: str          # UUID4
    status: SessionState
    text: str                # Sanitized + PII-masked text
    voice_id: str
    iteration: int
    max_iterations: int      # From config.security.max_retries
    wer: float | None
    is_approved: bool
    needs_human_review: bool
    audio_bytes: bytes
    agent_log: list[dict]
    error_message: str | None
```

| Операция | API | Complexity |
|----------|-----|-----------|
| Create | `_store.create(text, voice_id)` | O(1) |
| Get | `_store.get(session_id)` | O(1) |
| Update | `_store.update(session)` | O(1) |
| List | Нет (PoC) | — |
| Delete | Нет (GC при перезапуске) | — |

### Lifecycle

```
queued ──▶ processing ──▶ completed
                    └──▶ failed
```

**TTL**: нет (PoC). Planned: Redis TTL = 3600s (1 час).

---

## 2. GraphState — Pipeline Context

### Размер в памяти (типичный запрос)

| Поле | Примерный размер | Описание |
|------|------------------|----------|
| `text` | 1-5 KB | Входной текст |
| `ssml_markup` | 2-10 KB | JSON с сегментами |
| `audio_bytes` | 100 KB – 5 MB | Финальный WAV |
| `segment_audio[]` | 100 KB – 5 MB | Per-segment WAV |
| `errors[]` | 0.5-5 KB | Список ошибок |
| `agent_log[]` | 1-5 KB | Журнал действий |
| **Итого per session** | **~0.5 – 15 MB** | — |

### Serialization

```python
# Graph node → dict → next node
async def director_node(state: dict) -> dict:
    gs = GraphState.model_validate(state)    # dict → Pydantic
    gs = await run_director(gs, vllm)
    return gs.model_dump()                   # Pydantic → dict
```

Каждый узел: `model_validate` (deserialize) → business logic → `model_dump` (serialize).
Overhead: ~1ms на 1MB state.

---

## 3. Memory Policy

### Текущая (PoC)

| Политика | Реализация |
|----------|-----------|
| **No cross-session memory** | Каждый запрос — с нуля |
| **Intra-session only** | `GraphState` хранит всё состояние между итерациями |
| **No eviction** | Сессии в памяти до перезапуска |
| **No persistence** | При падении — данные потеряны |
| **No deduplication** | Одинаковые запросы — полный пересинтез |

### Планируемая (MAS-4)

| Память | Тип | Storage | Eviction |
|--------|-----|---------|----------|
| **Pronunciation memory** | Long-term | Redis hash | LRU, max 10K |
| **Segment cache** | Long-term | Redis binary | TTL 24h, max 1GB |
| **Repair log** | Long-term | SQLite | FIFO, max 100K |
| **Session state** | Short-term | Redis | TTL 1h |

---

## 4. Context Budget (LLM)

### vLLM Configuration

| Параметр | Значение | Описание |
|----------|---------|----------|
| `max-model-len` | 16384 tokens | Максимальное окно контекста |
| `max_tokens` (response) | 4096 tokens | Максимум в ответе |
| Total per request | ≤ 20480 tokens | Prompt + response |

### Budget per Agent

| Agent | System prompt | User message | Response | Total |
|-------|-------------|-------------|----------|-------|
| **Director** | ~500 tokens | ~100-1500 tokens (text) | ~500-2000 tokens (JSON) | ~1100-4000 |
| **Critic Judge** | ~400 tokens | ~200-600 tokens (target + transcript, truncated to 500 chars) | ~200-1000 tokens (JSON) | ~800-2000 |

### Truncation Guards

| Что | Лимит | Где |
|-----|-------|-----|
| User input text | 5000 chars | `SECURITY_MAX_TEXT_LENGTH` |
| Judge input: target text | 500 chars | In JUDGE_PROMPT |
| Judge input: timestamps | Max 10 | In JUDGE_PROMPT |
| Judge output: errors | Max 5 | In JUDGE_PROMPT |
| `<think>` blocks | Stripped | `_THINK_RE` regex |

### Context Overflow Protection

```
1. User text > 5000 chars → 400 Bad Request (sanitizer)
2. LLM context > 16384 tokens → vLLM truncates (silent)
3. LLM response > 4096 tokens → truncated (silent)
4. Judge prompt optimized: max 500 chars per field
```

---

## 5. Data Retention & Privacy

| Данные | Retention | Маскировка | Доступ |
|--------|-----------|-----------|--------|
| User text | Session lifetime | PII masked before pipeline | API, pipeline |
| Audio (intermediate) | Overwritten при retry | N/A | Pipeline only |
| Audio (final) | Session lifetime | N/A | API (GET /session/{id}/audio) |
| Agent log | Session lifetime | Нет PII (should be) | API, WebSocket |
| Prometheus metrics | App lifetime | Anonymized | /metrics endpoint |
| Raw logs | stdout/stderr | **⚠️ director_input_text leaks PII** | Log aggregator |
