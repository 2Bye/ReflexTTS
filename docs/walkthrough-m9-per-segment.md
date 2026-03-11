# Walkthrough: M9 — Per-Segment Critic Evaluation

## Проблема

До M9 Critic оценивал **весь аудио-файл целиком**, обрезая вход Judge до 500 символов. При ошибке в одном слове **пересинтезировался весь текст** — что при 3K символах означало ~5 минут × 3 итерации.

## Архитектурная схема

```
     До M9:                              После M9:
     ┌─────────────────┐                 ┌─────────────────┐
     │  Critic           │                 │  Critic           │
     │  ASR(all audio)   │                 │  ∀ segment i:     │
     │  Judge(500 chars) │                 │   ASR(seg[i])     │
     │  → all errors     │                 │   Judge(full)     │
     └────────┬─────────┘                 │  → seg errors     │
              │                            └────────┬─────────┘
     re-synth ALL                                   │
                                           re-synth ONLY failed
```

## Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `src/agents/schemas.py` | `segment_index: int` в `CriticError` |
| `src/orchestrator/state.py` | `segment_index` в `DetectedError`, `segment_audio`/`segment_approved` в `GraphState` |
| `src/agents/actor.py` | Per-segment WAV хранение, skip уже-approved сегментов при retry |
| `src/agents/critic.py` | Полная переработка: per-segment ASR+Judge, удалён truncation 500 chars |
| `src/agents/prompts.py` | `segment_index` в JSON-схеме промта Judge |
| `src/orchestrator/graph.py` | Роутинг учитывает per-segment статусы unapproved ошибок |

## Ключевые решения

### 1. Per-segment audio storage

Actor хранит WAV каждого сегмента в `state.segment_audio[i]`. На retry итерациях approved сегменты берутся из кеша — не пересинтезируются.

### 2. Per-segment evaluation

Critic оценивает **каждый сегмент отдельно** (если >1 сегмента). Каждый сегмент получает **полный текст** (без truncation), свой ASR pass и свою Judge оценку.

### 3. Обратная совместимость

Для коротких текстов (1 сегмент) — fallback на whole-audio evaluation (поведение до M9).

## Тесты

| Файл | Новые тесты | Описание |
|------|-------------|----------|
| `tests/unit/test_schemas.py` | +2 | `segment_index` default и explicit |
| `tests/unit/test_graph.py` | +3 | `segment_audio` defaults, roundtrip, `DetectedError.segment_index` |
| `tests/unit/test_actor.py` | +2 | `segment_audio` init, `segment_approved` reuse |

**Итого:** 164 passed, 1 pre-existing failure (`test_security_defaults` — ожидает 1000, config имеет 5000), ruff ✅, mypy ✅.

## Ожидаемый эффект (E2E)

| До M9 | После M9 |
|-------|----------|
| Critic обрезает до 500 chars | Judge получает полный текст каждого сегмента |
| Ошибка в 1 слове → пересинтез 100% | Ошибка в 1 слове → пересинтез только этого сегмента |
| 3K текст × 3 итерации = ~15 min | 3K текст × 3 итерации ≈ ~3-5 min (экономия на ре-синтезе) |
