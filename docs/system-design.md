# ReflexTTS — System Design

> Architectural document for the self-correcting speech synthesis PoC system.
> Version: 1.0 · Milestone 2 · 2026-03-25

---

## 1. Key Architectural Decisions

### 1.1 Multi-Agent Approach (Reflection Pattern)

| Decision | Rationale |
|----------|-----------|
| **4 agents** (Director, Actor, Critic, Editor) instead of monolithic pipeline | Each agent has single responsibility; easy debugging, replacement and scaling |
| **Centralized orchestrator** (LangGraph DAG) | Predictable control flow, deterministic routing, simplified debugging vs decentralized MAS |
| **Shared blackboard** (`GraphState`) | Data transfer between agents via one Pydantic model; all state in one place |
| **LLM-in-the-loop** for quality control | Critic Agent with two-phase evaluation (ASR + LLM Judge) — 100% semantic verification |
| **Segment-level operations** | Per-segment synthesis, evaluation and repair — minimizing GPU work on retry |

### 1.2 Inference Stack: Fully Local

| Decision | Rationale |
|----------|-----------|
| **Qwen3-8B AWQ 4-bit** via vLLM | OpenAI-compatible API, ~5 GB VRAM, JSON mode, reasoning support (`<think>`) |
| **CosyVoice3 0.5B** (Flow-Matching) | Bidirectional attention for inpainting, multi-speaker, instruct mode |
| **WhisperX large-v3** + Wav2Vec2 forced alignment | Word-level timestamps with confidence scores; required for precise error localization |
| **All models — local** | No cloud API dependency, latency control, no PII leakage to third parties |

### 1.3 Infrastructure Decisions

| Decision | Rationale |
|----------|-----------|
| **FastAPI** (async, WebSocket) | Async pipeline + real-time log streaming; embedded Web UI |
| **Pipeline in separate thread** with own event loop | Isolation from uvicorn event loop; deadlock prevention |
| **Queue + Worker thread** for pipeline | GPU-bound: requests queue up, processed sequentially |
| **Semaphore(4)** for TTS segments | Parallel synthesis of up to 4 segments within 1 pipeline |
| **Rate Limiter** | Sliding-window per-IP, 10 req/min (configurable) |
| **Redis session store** (optional) | TTL-based session management; `REDIS_USE_REDIS=true` |
| **In-memory session store** (default) | Quick start without external dependencies |
| **Pronunciation cache** | Cross-session phoneme hint cache (word+voice → hint) |
| **Segment audio cache** | Cross-session audio cache (SHA-256 keyed, WER=0 only) |

---

## 2. Module List and Roles

### 2.1 Module Map

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

### 2.2 Module Summary Table

| Module | Files | Role | Dependencies |
|--------|-------|------|-------------|
| **Director** | `agents/director.py`, `agents/prompts.py` | Text analysis → segments + emotions | VLLMClient |
| **Actor** | `agents/actor.py` | Segments → WAV audio (parallel synthesis) | TTSClient |
| **Critic** | `agents/critic.py`, `agents/prompts.py` | ASR → Judge → errors + WER | ASRClient, VLLMClient |
| **Editor** | `agents/editor.py` | Re-synthesis of unapproved segments | TTSClient |
| **Orchestrator** | `orchestrator/graph.py`, `orchestrator/state.py` | State machine + conditional routing | LangGraph, all agents |
| **Schemas** | `agents/schemas.py` | Data contracts (Pydantic) | — |
| **VLLMClient** | `inference/vllm_client.py` | LLM inference (Qwen3-8B) | AsyncOpenAI |
| **TTSClient** | `inference/tts_client.py` | TTS inference (CosyVoice3) | httpx |
| **ASRClient** | `inference/asr_client.py` | ASR inference (WhisperX) | httpx |
| **ModelRegistry** | `inference/model_registry.py` | Lifecycle: init → health → shutdown | all clients |
| **Audio Utils** | `audio/*.py` | Alignment, masking, crossfade, convergence | numpy |
| **Security** | `security/*.py` | Sanitization, PII, voice whitelist | regex |
| **API** | `api/app.py`, `api/schemas.py`, `api/sessions.py`, `api/rate_limiter.py`, `api/redis_store.py` | REST + WebSocket + Web UI + rate limiting | FastAPI |
| **Monitoring** | `monitoring/__init__.py`, `monitoring/tracing.py` | Prometheus metrics + OpenTelemetry tracing | opentelemetry-sdk |
| **Caches** | `agents/pronunciation_cache.py`, `agents/segment_cache.py` | Cross-session hint + audio caches | — |
| **Config** | `config.py` | Pydantic Settings from env | pydantic-settings |
| **Logging** | `log.py` | structlog: JSON (prod) / console (dev) | structlog |

---

## 3. Main Workflow

### 3.1 Execution Flow

```
POST /synthesize { text, voice_id }
  │
  ├── 1. Input Sanitization     → 10 regex patterns, max_length=5000
  ├── 2. PII Masking             → email, phone, card, passport → [MASK]
  ├── 3. Voice Validation        → whitelist check (3 speakers)
  ├── 4. Rate Limiting           → sliding-window per-IP, 429 if exceeded
  ├── 5. Session Creation        → UUID, state=queued
  ├── 6. Enqueue pipeline        → queue.Queue + daemon worker thread
  │     │
  │     ▼
  │   ┌──────────────────────────────────────────────────┐
  │   │  iteration = 0                                    │
  │   │                                                   │
  │   │  Director(Qwen3-8B)                               │
  │   │    text → chat_json() → DirectorOutput            │
  │   │    → segments[] + emotions + phoneme_hints        │
  │   │    → if iteration>0: _apply_hotfix_hints()        │
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
  │   │  Editor(CosyVoice3) [if route=editor]             │
  │   │    failed_segs = _get_failed_segments()           │
  │   │    ∀ failed segment: tts.synthesize() → new WAV   │
  │   │    _rebuild_combined_audio()                      │
  │   │    convergence_score()                            │
  │   │    → back to Critic                               │
  │   └──────────────────────────────────────────────────┘
  │
  ├── 6. Session Update          → status=completed, audio_bytes
  └── 7. Response/WebSocket      → agent_log stream
```

### 3.2 Routing Logic (detailed)

| Condition | Route | Action |
|-----------|-------|--------|
| `is_approved == True` | `approved` | → END, audio ready |
| `iteration >= max_retries` | `needs_human_review` | → END, escalation flag |
| All unapproved segment errors have `can_hotfix=True` | `hotfix` | → Director with phoneme hints |
| Errors with `can_hotfix=False` exist | `editor` | → Editor, segment re-synthesis |

### 3.3 Typical Scenarios

| Scenario | Iterations | WER | Size |
|----------|-----------|-----|------|
| Short text (~50 chars), clean synthesis | 1 | 0.000 | ~150 KB |
| Medium text (~500 chars), 1 error | 2 | 0.000 | ~2 MB |
| Long text (~800 chars), complex words | 2-3 | 0.000 | ~2.5 MB |
| Maximum complexity, max_retries | 5 | >0 | — (human review) |

---

## 4. State / Memory / Context Handling

### 4.1 GraphState — Central Structure

`GraphState` (Pydantic `BaseModel`) — the single object passed between LangGraph nodes. Contains:

| Group | Fields | Description |
|-------|--------|-------------|
| **Input** | `text`, `voice_id`, `trace_id` | User input data |
| **Director** | `ssml_markup`, `tts_instruct` | Structured synthesis instructions |
| **Actor** | `audio_bytes`, `sample_rate`, `segment_audio[]`, `segment_approved[]` | Audio data |
| **Critic** | `transcript`, `word_timestamps[]`, `errors[]`, `wer`, `is_approved` | Evaluation results |
| **Control** | `iteration`, `max_retries`, `needs_human_review`, `convergence_score` | Flow control |
| **Log** | `agent_log[]` | Agent action journal |

### 4.2 Memory Policy

| Aspect | Implementation | Limitation |
|--------|---------------|------------|
| **Session memory** | In-memory `SessionStore` (default) or `RedisSessionStore` (`REDIS_USE_REDIS=true`) | Redis: TTL 1h, survives restart |
| **Cross-session pronunciation** | `PronunciationCache` — word+voice → phoneme hint | ✅ In-memory, threshold=2 successes |
| **Cross-session audio** | `SegmentCache` — SHA-256(text+voice+emotion) → WAV | ✅ In-memory, TTL 24h, WER=0 only |
| **Agent memory** | No persistent memory | Agents stateless between requests |
| **Ephemeral data** | WAV, intermediate results — in RAM | Deleted after session completion |
| **Logging** | Anonymized metadata only | ✅ PII removed from logs |

### 4.3 Context Budget

| Context | Budget | Rationale |
|---------|--------|-----------|
| vLLM `max-model-len` | **16384 tokens** | Sufficient for long Director + Judge prompts |
| Director prompt + text | ~2000-3000 tokens | System prompt + user text up to 5000 chars |
| Judge prompt + target + transcript | ~1000-2000 tokens | Truncated to 500 chars + max 10 timestamps |
| `max_tokens` response | **4096 tokens** | Maximum for JSON response |

---

## 5. Retrieval Pipeline

### 5.1 Current State

The current PoC system has **no retrieval pipeline**. The system operates in generative-only mode:

- Director generates segmentation based on prompt and input text
- Critic evaluates quality via ASR + LLM-Judge
- No external knowledge bases, embedding stores or vector databases

### 5.2 Retrieval-like Mechanisms

| Mechanism | Description | Status |
|----------|----------|--------|
| **Phoneme hint lookup** | Director can use `phoneme_hints` from previous Critic errors | ✅ Implemented (intra-session) |
| **Segment cache (intra)** | Actor reuses `segment_audio[i]` if `segment_approved[i]=True` | ✅ Implemented (intra-session) |
| **Voice lookup** | `VOICE_MAP` in TTSClient for `voice_id → speaker name` mapping | ✅ Implemented |
| **Pronunciation memory** | Cross-session cache (word, voice) → phoneme_hint, threshold=2 successes | ✅ Implemented (`pronunciation_cache.py`) |
| **Segment audio cache** | SHA-256(text + voice + emotion) → WAV bytes, TTL 24h, WER=0 only | ✅ Implemented (`segment_cache.py`) |

### 5.3 Planned Retrieval Pipeline (MAS-4)

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Director    │────▶│ Pronunciation    │────▶│ Redis/SQLite │
│              │     │ Memory Retrieval │     │ Key-Value    │
│ "Which hint  │     │ (word, voice_id) │     │ Store        │
│  worked      │     │  → phoneme_hint  │     │              │
│  before?"    │     │  → success_rate  │     │              │
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

## 6. Tool / API Integrations

### 6.1 Internal Model APIs

| Service | Protocol | Endpoint | Used by | Timeout | Retry |
|---------|----------|----------|---------|---------|-------|
| **vLLM** (Qwen3-8B) | OpenAI-compat HTTP | `:8055/v1/chat/completions` | Director, Critic | 300s | 5× exponential |
| **CosyVoice3** (0.5B) | HTTP/REST | `:9880/tts` | Actor, Editor | 60s | — |
| **WhisperX** (large-v3) | HTTP/REST | `:9881/transcribe` | Critic | 60s | — |

### 6.2 External APIs / Side Effects

| API | Purpose | Side Effects | Protection |
|-----|---------|-------------|-----------|
| **Redis** (`:8056`) | Session storage (optional, `REDIS_USE_REDIS=true`) | Read/write TTL-based sessions | maxmemory=256MB, allkeys-lru |
| No cloud APIs | All inference is local | — | — |

### 6.3 Contracts (details in `docs/specs/tools-apis.md`)

**VLLMClient:**
- `chat(system_prompt, user_message) → str`
- `chat_json(system_prompt, user_message, response_model) → T`
- Fallback: `<think>` stripping → JSON parse → brace extraction → error

**TTSClient:**
- `synthesize(text, voice_id, instruct?) → AudioResult(waveform, sample_rate)`
- `clone_voice(text, ref_audio, ref_text) → AudioResult`
- Guard: voice whitelist validation before call

**ASRClient:**
- `transcribe(audio_bytes, sample_rate) → TranscriptionResult(text, word_timestamps[], language)`

---

## 7. Failure Modes, Fallback and Guardrails

### 7.1 Failure Modes

| Failure | Probability | Impact | Detection | Mitigation |
|---------|------------|--------|-----------|------------|
| **Infinite correction loop** | ⚠️ Medium | OOM, GPU waste | `iteration >= max_retries` | Hard limit `MAX_RETRIES=5`, escalation to human review |
| **vLLM connection timeout** | ⚠️ Medium | Pipeline stall | `APIConnectionError` / `APITimeoutError` | 5× retry with exponential backoff (2^n sec); 300s hard timeout |
| **JSON parse failure** (Qwen3 `<think>`) | ⚠️ Medium | Pipeline crash | `json.JSONDecodeError` | 3-step fallback: strip `<think>` → parse → brace extraction |
| **CosyVoice assertion** (instruct token) | Low | TTS failure | `AssertionError` | Auto-append `<\|endofprompt\|>` in server.py |
| **Pipeline timeout** (300s) | Low | Session stuck | `asyncio.wait_for` timeout | Graceful failure → `status=failed`, queue release |
| **Prompt injection** | ⚠️ Medium | Unsafe LLM behavior | 10 compiled regex patterns | `sanitize_input()` → reject with 400 |
| **PII leakage** | ⚠️ Medium | Privacy violation | Regex: email/phone/card/passport/INN/IP | `mask_pii()` → `[EMAIL_1]`, `[PHONE_1]` |
| **Voice spoofing** | Low (PoC) | Deepfake risk | Voice whitelist check | Only 3 predefined voices; zero-shot cloning disabled |
| **GPU OOM** | Low | Container crash | Docker healthcheck | Queue-based pipeline — 1 pipeline at a time; GPU memory utilization=0.7 |
| **Event loop deadlock** | Low | App hangs | Health check fails | Pipeline in separate Thread with own event loop |

### 7.2 Guardrails

| Guardrail | Implementation | Configuration |
|-----------|---------------|--------------|
| **Text length** | `max_length=5000` in sanitizer | `SECURITY_MAX_TEXT_LENGTH` |
| **Retry limit** | `max_retries=5` | `SECURITY_MAX_RETRIES` |
| **Pipeline concurrency** | `queue.Queue` + single worker thread | Sequential processing |
| **Rate limiting** | Sliding-window per-IP, 10 req/min | `API_RATE_LIMIT_PER_MINUTE` |
| **TTS concurrency** | `asyncio.Semaphore(4)` | `max_concurrency` param |
| **Pipeline timeout** | `asyncio.wait_for(300s)` | Hardcoded |
| **Voice whitelist** | 3 speakers | `SECURITY_WHITELISTED_VOICES` |
| **Unknown emotion → neutral** | Validator in `Segment` | `_fallback_unknown_emotion()` |
| **Empty JSON fallback** | Brace extraction in VLLMClient | Automatic |
| **WER threshold for escalation** | 0.15 | `SECURITY_WER_THRESHOLD_FOR_HUMAN_REVIEW` |
| **Judge input truncation** | 500 chars + 10 timestamps | In prompt |

### 7.3 Escalation Path

```
Critic not approved
  └── iteration < max_retries?
        ├── YES: hotfix/editor loop continues
        └── NO: mark_human_review()
                  → needs_human_review = True
                  → agent_log: "escalated"
                  → status = completed (with warning)
                  → UI: "Approve Audio with Errors" / "Edit Original Text"
```

---

## 8. Technical and Operational Constraints

### 8.1 Latency

| Operation | p50 | p95 | Bottleneck |
|-----------|-----|-----|-----------|
| Director (Qwen3-8B, JSON) | 2-5s | 8-15s | GPU inference, token generation |
| Actor (CosyVoice3, 1 segment) | 1-3s | 5-8s | TTS inference, audio encoding |
| Actor (CosyVoice3, 6 segments, parallel) | 3-6s | 10-15s | GPU parallelism (Semaphore 4) |
| Critic Phase 1 (WhisperX ASR) | 1-2s | 3-5s | ASR inference |
| Critic Phase 2 (Judge, JSON) | 2-4s | 6-10s | GPU inference |
| Editor (1 segment re-synth) | 1-3s | 5-8s | TTS inference |
| **Full pipeline (1 iteration, short text)** | **8-15s** | **20-30s** | — |
| **Full pipeline (3 iterations, long text)** | **30-60s** | **90-120s** | — |
| **Hard timeout** | — | **300s** | — |

> **RTF (Real-Time Factor)** for 10s audio: p50 ≈ 1.5-3.0, p95 ≈ 3.0-6.0

### 8.2 Cost (PoC: self-hosted GPU)

| Resource | Consumption | Cost |
|----------|------------|------|
| GPU 1 (A4000 16GB) | vLLM: ~5 GB VRAM, utilization 70% | Hardware amortization |
| GPU 2 (A4000 16GB) | CosyVoice3 (~2 GB) + WhisperX (~3 GB) | Hardware amortization |
| CPU | FastAPI + orchestrator | Minimal |
| RAM | Session store + intermediate audio | ~1-2 GB per session peak |
| Storage | Redis 256MB + session WAV | Ephemeral, TTL 1h |

**Cost per request (inference tokens):**
- Director: ~1000-2000 output tokens
- Judge: ~500-1000 output tokens per segment
- At 3 iterations × 6 segments: ~10K-20K tokens total
- Self-hosted: $0 marginal (with GPU available)

### 8.3 Reliability

| Metric | Current | Target |
|--------|---------|--------|
| Pipeline success rate | ≈ 95% (PoC) | > 99% |
| WER on approved | 0.000 (observed) | < 0.01 |
| Human acceptance rate | — (no data) | > 95% |
| Avg iterations to converge | 1-3 | < 2.5 |
| Uptime | — | 99.9% (with docker restart) |

### 8.4 Limits & Quotas

| Parameter | Value | Configuration |
|-----------|-------|--------------|
| Max text length | 5000 chars | `SECURITY_MAX_TEXT_LENGTH` |
| Max retries | 5 | `SECURITY_MAX_RETRIES` |
| Max concurrent pipelines | 1 (queued) | `queue.Queue` + worker |
| Max concurrent TTS segments | 4 | `Semaphore(4)` |
| Pipeline hard timeout | 300s | Hardcoded |
| Session TTL | 1h | `REDIS_SESSION_TTL_SECONDS` |
| API rate limit | 10 req/min | `API_RATE_LIMIT_PER_MINUTE` |
| Max JSON response tokens | 4096 | `VLLM_MAX_TOKENS` |
| vLLM context window | 16384 tokens | `max-model-len` |

---

## 9. Detailed Specification References

| Specification | File |
|--------------|------|
| Retriever / Retrieval | [docs/specs/retriever.md](specs/retriever.md) |
| Tools / APIs | [docs/specs/tools-apis.md](specs/tools-apis.md) |
| Memory / Context | [docs/specs/memory-context.md](specs/memory-context.md) |
| Agent / Orchestrator | [docs/specs/agent-orchestrator.md](specs/agent-orchestrator.md) |
| Serving / Config | [docs/specs/serving-config.md](specs/serving-config.md) |
| Observability / Evals | [docs/specs/observability-evals.md](specs/observability-evals.md) |
| Diagrams (C4 + Flow) | [docs/diagrams/](diagrams/) |
