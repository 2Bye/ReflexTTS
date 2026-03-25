# Data Flow Diagram — ReflexTTS

> Как данные проходят через систему, что хранится, что логируется.

## End-to-End Data Flow

```mermaid
flowchart TB
    subgraph Input["📥 Input"]
        UserText["User Text<br/>(1-5000 chars)"]
        VoiceID["voice_id<br/>(speaker_1/2/3)"]
    end

    subgraph Security["🔒 Security Layer"]
        direction TB
        Sanitize["Input Sanitizer<br/>10 regex patterns<br/>max_length check"]
        PIIMask["PII Masker<br/>email→[EMAIL_1]<br/>phone→[PHONE_1]<br/>card→[CARD_1]"]
        VoiceCheck["Voice Whitelist<br/>3 allowed speakers"]
    end

    subgraph Session["💾 Session Store"]
        SessionCreate["Create Session<br/>UUID, status=queued"]
        SessionUpdate["Update Session<br/>agent_log, wer, audio"]
    end

    subgraph Director["🎬 Director"]
        direction TB
        DPrompt["System Prompt<br/>+ sanitized text"]
        DLLM["Qwen3-8B<br/>chat_json()"]
        DOutput["DirectorOutput<br/>segments[]: text,emotion<br/>pause_before_ms<br/>phoneme_hints[]"]
        DHotfix["_apply_hotfix_hints<br/>inject phoneme → text"]
    end

    subgraph Actor["🎤 Actor"]
        direction TB
        AParallel["asyncio.gather<br/>Semaphore(4)"]
        ATTS["CosyVoice3<br/>synthesize(text,<br/>voice_id, instruct)"]
        AEncode["WAV Encode<br/>16-bit PCM<br/>24000 Hz"]
        AConcat["Concatenate<br/>pause + segments"]
    end

    subgraph Critic["🔍 Critic"]
        direction TB
        CASR["WhisperX ASR<br/>transcribe(wav)<br/>→ text + timestamps"]
        CJudge["Qwen3-8B Judge<br/>chat_json(target<br/>vs transcript)"]
        COutput["CriticOutput<br/>errors[], wer<br/>is_approved<br/>segment_approved[]"]
    end

    subgraph Editor["✏️ Editor"]
        direction TB
        EFind["_get_failed_segments<br/>from segment_approved<br/>+ errors"]
        ERegen["CosyVoice3<br/>re-synth failed<br/>segments only"]
        ERebuild["_rebuild_combined<br/>_audio()"]
        EConv["convergence_score<br/>0.5(1-WER)+0.3SECS<br/>+0.2(PESQ/4.5)"]
    end

    subgraph Output["📤 Output"]
        AudioWAV["audio_bytes<br/>WAV 16-bit PCM"]
        AgentLog["agent_log[]<br/>agent, action, detail"]
        Metrics["Prometheus metrics<br/>latency, WER, iterations"]
    end

    %% Connections
    UserText --> Sanitize
    VoiceID --> VoiceCheck
    Sanitize --> PIIMask
    PIIMask --> SessionCreate
    VoiceCheck --> SessionCreate

    SessionCreate --> DPrompt
    DPrompt --> DLLM
    DLLM --> DOutput
    DOutput --> AParallel

    AParallel --> ATTS
    ATTS --> AEncode
    AEncode --> AConcat

    AConcat --> CASR
    CASR --> CJudge
    CJudge --> COutput

    COutput -->|"approved"| AudioWAV
    COutput -->|"hotfix"| DHotfix
    DHotfix --> DPrompt
    COutput -->|"editor"| EFind
    EFind --> ERegen
    ERegen --> ERebuild
    ERebuild --> EConv
    EConv --> CASR

    SessionCreate --> SessionUpdate
    SessionUpdate --> AgentLog
    SessionUpdate --> Metrics
    AudioWAV --> SessionUpdate

    style Input fill:#1a1a2e,stroke:#6c5ce7,color:#e8e8f0
    style Security fill:#1a1a2e,stroke:#ff6b6b,color:#e8e8f0
    style Director fill:#1a1a2e,stroke:#6c5ce7,color:#e8e8f0
    style Actor fill:#1a1a2e,stroke:#00d2a0,color:#e8e8f0
    style Critic fill:#1a1a2e,stroke:#feca57,color:#e8e8f0
    style Editor fill:#1a1a2e,stroke:#a29bfe,color:#e8e8f0
    style Output fill:#1a1a2e,stroke:#00d2a0,color:#e8e8f0
    style Session fill:#1a1a2e,stroke:#8888a0,color:#e8e8f0
```

## Data Storage & Lifecycle

### Что хранится

| Данные | Где | Время жизни | Формат |
|--------|-----|------------|--------|
| `GraphState` | RAM (thread local) | Время выполнения pipeline | Pydantic model → dict |
| `SessionState` | In-memory dict (PoC) | До перезапуска сервера | Python dataclass |
| `audio_bytes` | `SessionState.audio_bytes` | До GC / перезапуска | WAV bytes |
| `segment_audio[]` | `GraphState` (RAM) | Время выполнения pipeline | WAV bytes per segment |
| `agent_log[]` | `SessionState` + WebSocket | До перезапуска / export | list[dict] |
| Prometheus metrics | `MetricsRegistry` (RAM) | До перезапуска | Counter/Gauge/Histogram |

### Что логируется

| Уровень | Что | PII | Пример |
|---------|-----|-----|--------|
| INFO | Agent actions | ❌ Нет | `director_done segments=3 instruct="Speak with happy tone"` |
| INFO | Pipeline lifecycle | ❌ Нет | `pipeline_completed wer=0.0 iterations=2 audio_size_kb=150` |
| WARNING | Errors, retries | ❌ Нет | `injection_detected patterns=["role_override"]` |
| WARNING | Routing decisions | ❌ Нет | `route_editor iteration=1 failed_segments=[1,3]` |
| ERROR | Failures | ❌ Нет | `pipeline_failed error="VLLMConnectionError"` |
| ⚠️ INFO | Director input text | ✅ **Да** | `director_input_text text="..."` — **должно быть убрано** |
| DEBUG | Raw LLM responses | Возможно | `vllm_raw_response raw_preview=...` |

> **Known issue**: `director_input_text` логирует сырой текст пользователя. Необходимо убрать или маскировать.

### Что НЕ хранится

- ❌ Промежуточные отклонённые аудио (overwritten при retry)
- ❌ Полные LLM промпты после выполнения
- ❌ Embedding'и аудио/текста (нет retrieval layer)
- ❌ Данные между сессиями (нет persistent storage в PoC)

## Data Transformation Pipeline

```
                    STRING            PYDANTIC            DICT              BYTES
User text ─────────────▶ sanitized ───────▶ GraphState ──────▶ state dict ──────▶ WAV
  │                        text                │                    │              │
  │                         │                  │                    │              │
  │    sanitize_input()     │   model_dump()   │   graph.invoke()   │   actor      │
  │    mask_pii()           │   model_validate │   director_node    │   encode_wav │
  │                         │                  │   actor_node       │              │
  │                         │                  │   critic_node      │              │
  │                         │                  │   editor_node      │              │
  │                         │                  │                    │              │
  ▼                         ▼                  ▼                    ▼              ▼

Types:  str             str           GraphState (Pydantic)   dict           bytes
Size:   1-5000 chars    1-5000 chars  ~20 fields              ~20 keys       ~100KB-5MB
```
