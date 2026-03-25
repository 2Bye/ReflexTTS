# C4 Container Diagram — ReflexTTS

> Уровень 2: frontend/backend, orchestrator, вreiever, tool layer, storage, observability.

```mermaid
C4Container
    title ReflexTTS — Container Diagram

    Person(user, "User", "REST / WebSocket / Web UI")

    Container_Boundary(app_boundary, "ReflexTTS Application") {
        Container(api, "FastAPI Server", "Python, Uvicorn", "REST API + WebSocket + встроенный Web UI. Port :8081")
        Container(security, "Security Layer", "Python", "Input Sanitizer (10 regex), PII Masker, Voice Whitelist")
        Container(orchestrator, "LangGraph Orchestrator", "Python, LangGraph", "StateGraph: Director → Actor → Critic → route → [Editor/Hotfix/End]")
        Container(agents, "Agent Layer", "Python", "Director, Actor, Critic, Editor — бизнес-логика агентов")
        Container(audio, "Audio Utils", "Python, NumPy", "Alignment, Masking, Crossfade, Convergence metrics")
        Container(sessions, "Session Store", "Python, In-Memory", "UUID → SessionState (PoC). Planned: Redis")
        Container(monitoring_mod, "Metrics Module", "Python", "Counter, Gauge, Histogram → Prometheus text format")
    }

    Container_Boundary(gpu_boundary, "GPU Services") {
        ContainerDb(vllm, "vLLM Server", "Qwen3-8B AWQ", "OpenAI-compat API :8055. GPU 1, ~5GB VRAM")
        ContainerDb(cosyvoice, "CosyVoice3 Service", "Fun-CosyVoice3-0.5B", "HTTP REST :9880. GPU 2, ~2GB VRAM")
        ContainerDb(whisperx, "WhisperX Service", "large-v3 + Wav2Vec2", "HTTP REST :9881. GPU 2, ~3GB VRAM")
    }

    ContainerDb(redis, "Redis", "Redis 7 Alpine", "Port :8056. Session TTL, maxmem 256MB")

    Rel(user, api, "HTTP / WebSocket", "JSON, WAV, agent log stream")
    Rel(api, security, "Validates input", "")
    Rel(api, orchestrator, "graph.invoke(state)", "threading.Thread + own event loop")
    Rel(api, sessions, "CRUD", "")
    Rel(api, monitoring_mod, "export()", "")

    Rel(orchestrator, agents, "run_director/actor/critic/editor(state)", "")
    Rel(agents, vllm, "chat_json()", "OpenAI API, JSON mode")
    Rel(agents, cosyvoice, "synthesize()", "HTTP, WAV response")
    Rel(agents, whisperx, "transcribe()", "HTTP, multipart audio")
    Rel(agents, audio, "convergence_score()", "")

    Rel(sessions, redis, "Planned", "Redis key-value")
```

## Детали контейнеров

### Application Layer

| Контейнер | Технология | Port | Роль |
|-----------|-----------|------|------|
| **FastAPI Server** | Python, Uvicorn | :8081 | REST endpoints + WebSocket + embedded Web UI |
| **Security Layer** | Python, regex | — | Input validation, PII masking, voice access control |
| **LangGraph Orchestrator** | Python, LangGraph | — | DAG execution, conditional routing, state management |
| **Agent Layer** | Python | — | 4 агента: бизнес-логика pipeline |
| **Audio Utils** | Python, NumPy | — | Low-level audio processing |
| **Session Store** | Python, dict | — | In-memory session management (PoC) |
| **Metrics Module** | Python | — | Custom Prometheus metrics export |

### GPU Services

| Сервис | Container | GPU | VRAM | Port |
|--------|-----------|-----|------|------|
| vLLM (Qwen3-8B AWQ) | `reflex-vllm` | GPU 1 | ~5 GB | :8055 |
| CosyVoice3 (0.5B) | `reflex-cosyvoice` | GPU 2 | ~2 GB | :9880 |
| WhisperX (large-v3) | `reflex-whisperx` | GPU 2 | ~3 GB | :9881 |

### Storage

| Сервис | Container | Port | Config |
|--------|-----------|------|--------|
| Redis | `reflex-redis` | :8056 | maxmemory=256MB, TTL=1h |

### Коммуникация

```
User ──HTTP──▶ FastAPI ──Thread──▶ LangGraph ──async──▶ Agents
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                    vLLM :8055    CosyVoice :9880  WhisperX :9881
                                    (OpenAI API)     (HTTP)           (HTTP)
```
