# Spec: Memory / Context

> Session state, memory policy, context budget.

---

## 1. Session State

### SessionStore (In-Memory / Redis)

**Files:** `src/api/sessions.py`, `src/api/redis_store.py`

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
    queue_position: int | None       # Position in pipeline queue
```

**Backend selection** (`create_session_store(config)`):
- `REDIS_USE_REDIS=false` (default) → `SessionStore` (in-memory dict)
- `REDIS_USE_REDIS=true` → `RedisSessionStore` (Redis, TTL-based)

| Operation | API | Complexity |
|-----------|-----|-----------|
| Create | `_store.create(text, voice_id)` | O(1) |
| Get | `_store.get(session_id)` | O(1) |
| Update | `_store.update(session)` | O(1) |
| List | None (PoC) | — |
| Delete | None (GC on restart) | — |

### Lifecycle

```
queued ──▶ processing ──▶ completed
                    └──▶ failed
```

**TTL**: In-memory: none. Redis: `REDIS_SESSION_TTL_SECONDS` = 3600s (1 hour).

---

## 2. GraphState — Pipeline Context

### Size in Memory (typical request)

| Field | Approximate Size | Description |
|-------|------------------|-------------|
| `text` | 1-5 KB | Input text |
| `ssml_markup` | 2-10 KB | JSON with segments |
| `audio_bytes` | 100 KB – 5 MB | Final WAV |
| `segment_audio[]` | 100 KB – 5 MB | Per-segment WAV |
| `errors[]` | 0.5-5 KB | Error list |
| `agent_log[]` | 1-5 KB | Agent action journal |
| **Total per session** | **~0.5 – 15 MB** | — |

### Serialization

```python
# Graph node → dict → next node
async def director_node(state: dict) -> dict:
    gs = GraphState.model_validate(state)    # dict → Pydantic
    gs = await run_director(gs, vllm)
    return gs.model_dump()                   # Pydantic → dict
```

Each node: `model_validate` (deserialize) → business logic → `model_dump` (serialize).
Overhead: ~1ms per 1MB state.

---

## 3. Memory Policy

### Current

| Policy | Implementation |
|--------|---------------|
| **Cross-session pronunciation** | `PronunciationCache` — word+voice → phoneme hint |
| **Cross-session audio** | `SegmentCache` — SHA-256(text+voice+emotion) → WAV |
| **Intra-session only** | `GraphState` stores all state between iterations |
| **Eviction** | In-memory: none. Redis: TTL. Caches: LRU |
| **Persistence** | In-memory: none. Redis: yes |
| **Deduplication** | `SegmentCache`: SHA-256(text+voice+emotion), WER=0 only |

### Future (MAS-4)

| Memory | Type | Storage | Eviction | Status |
|--------|------|---------|----------|--------|
| **Pronunciation memory** | Long-term | In-memory (Redis planned) | LRU, max 10K | ✅ `pronunciation_cache.py` |
| **Segment cache** | Long-term | In-memory (Redis planned) | TTL 24h, LRU 1K | ✅ `segment_cache.py` |
| **Repair log** | Long-term | SQLite | FIFO, max 100K | ⬜ Planned |
| **Session state** | Short-term | In-memory / Redis | TTL 1h | ✅ `sessions.py` / `redis_store.py` |

---

## 4. Context Budget (LLM)

### vLLM Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max-model-len` | 16384 tokens | Maximum context window |
| `max_tokens` (response) | 4096 tokens | Maximum in response |
| Total per request | ≤ 20480 tokens | Prompt + response |

### Budget per Agent

| Agent | System prompt | User message | Response | Total |
|-------|-------------|-------------|----------|-------|
| **Director** | ~500 tokens | ~100-1500 tokens (text) | ~500-2000 tokens (JSON) | ~1100-4000 |
| **Critic Judge** | ~400 tokens | ~200-600 tokens (target + transcript, truncated to 500 chars) | ~200-1000 tokens (JSON) | ~800-2000 |

### Truncation Guards

| What | Limit | Where |
|------|-------|-------|
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

| Data | Retention | Masking | Access |
|------|-----------|---------|--------|
| User text | Session lifetime | PII masked before pipeline | API, pipeline |
| Audio (intermediate) | Overwritten on retry | N/A | Pipeline only |
| Audio (final) | Session lifetime | N/A | API (GET /session/{id}/audio) |
| Agent log | Session lifetime | No PII | API, WebSocket |
| Prometheus metrics | App lifetime | Anonymized | /metrics endpoint |
| Raw logs | stdout/stderr | ✅ **PII removed** (`director_input` logs only text_length) | Log aggregator |
