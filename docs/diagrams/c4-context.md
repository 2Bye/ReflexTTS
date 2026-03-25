# C4 Context Diagram — ReflexTTS

> Уровень 1: система, пользователь, внешние сервисы и границы.

```mermaid
C4Context
    title ReflexTTS — System Context Diagram

    Person(user, "User / Client", "Отправляет текст для синтеза, получает аудио WAV")

    System(reflexTTS, "ReflexTTS", "Self-correcting TTS pipeline с 4 агентами (Director → Actor → Critic → Editor). Локальный GPU inference.")

    System_Ext(browser, "Web Browser", "Встроенный Web UI для интерактивного синтеза")

    System_Ext(monitoring, "Prometheus / Grafana", "Сбор метрик: latency, WER, iterations, errors")

    Rel(user, reflexTTS, "REST API / WebSocket", "POST /synthesize, GET /session, WS /ws")
    Rel(browser, reflexTTS, "HTTP + WebSocket", "Web UI, real-time agent log")
    Rel(reflexTTS, monitoring, "HTTP /metrics", "Prometheus exposition format")
```

## Текстовое описание

### Акторы

| Актор | Описание |
|-------|----------|
| **User / Client** | Человек или программа, отправляющая текст через REST API или Web UI |
| **Web Browser** | Встроенный UI (HTML/JS/CSS) для интерактивного доступа |
| **Prometheus / Grafana** | Внешний мониторинг, скрейпит `/metrics` |

### Система

**ReflexTTS** — self-correcting text-to-speech pipeline:
- Принимает текст + voice_id
- Проводит через 4 агента (Director → Actor → Critic → Editor)
- Итеративно исправляет ошибки произношения
- Возвращает WAV аудио с WER ≈ 0

### Границы

```
┌─────────────────────────────────────────────┐
│                Trust Boundary               │
│  ┌────────────────────────────────────────┐  │
│  │         ReflexTTS System               │  │
│  │  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │ FastAPI   │  │ LangGraph        │   │  │
│  │  │ + Web UI  │  │ Orchestrator     │   │  │
│  │  └──────────┘  └──────────────────┘   │  │
│  │  ┌──────────────────────────────────┐ │  │
│  │  │ GPU Services (vLLM, CosyVoice,  │ │  │
│  │  │ WhisperX) — all local            │ │  │
│  │  └──────────────────────────────────┘ │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ⛔ Нет внешних облачных API                 │
│  ⛔ Нет исходящих запросов с PII             │
│  ✅ Все данные остаются внутри trust boundary│
└─────────────────────────────────────────────┘
```

### Ключевые свойства

- **Полностью локальная система** — нет зависимостей от облачных LLM/TTS/ASR API
- **Единственный внешний интерфейс** — Prometheus scraping (read-only, no PII)
- **PII boundary** — маскировка происходит до входа в pipeline
