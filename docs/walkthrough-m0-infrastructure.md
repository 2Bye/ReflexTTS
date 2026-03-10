# Milestone 0: Инфраструктура

## Результат
✅ Lint ✅ Mypy ✅ 12 тестов

## Структура проекта

```
src/
├── config.py              # Pydantic Settings (все компоненты)
├── log.py                 # structlog (JSON/console, PII-free)
├── agents/                # M2+
├── orchestrator/
│   └── state.py           # GraphState, DetectedError, AgentLogEntry
├── inference/             # M1
├── security/              # M4
├── audio/                 # M3
└── api/
    └── app.py             # FastAPI: /health, /voices
```

## Ключевые файлы

| Файл | Назначение |
|------|-----------|
| `pyproject.toml` | Зависимости, ruff, mypy strict, pytest |
| `src/config.py` | Pydantic Settings для всех компонентов (env-prefixed) |
| `src/log.py` | structlog: JSON (prod) / console (dev) |
| `src/orchestrator/state.py` | `GraphState`, `DetectedError`, `AgentLogEntry` |
| `src/api/app.py` | FastAPI factory с health check |
| `docker/docker-compose.yml` | vLLM + Redis + App сервисы |
| `docker/Dockerfile.app` | Multi-stage, non-root, health checks |
| `.github/workflows/ci.yml` | CI: lint → type-check → security → tests |
| `.pre-commit-config.yaml` | Ruff, Mypy, Bandit, Detect Secrets |
| `.env.example` | Шаблон переменных окружения |
