# Spec: Retriever / Retrieval

> Источники данных, индексы, поиск, реранкинг, ограничения.

---

## Текущее состояние: Retrieval-free

В PoC-версии ReflexTTS **нет классического retrieval-контура** (vector store, embedding search, RAG). Система работает в generative-only режиме.

## Retrieval-подобные механизмы (реализованные)

### 1. Intra-session Phoneme Lookup

| Параметр | Значение |
|----------|---------|
| **Источник** | `GraphState.errors[].hotfix_hint` (от Critic) |
| **Индекс** | Нет — прямая итерация по `errors[]` |
| **Поиск** | `word_expected in segment.text` (substring match) |
| **Результат** | Инъекция phoneme hint в текст сегмента |
| **Scope** | Только текущая сессия (intra-session) |

```python
# director.py → _apply_hotfix_hints()
for error in state.errors:
    if error.can_hotfix and error.hotfix_hint:
        for segment in output.segments:
            if error.word_expected in segment.text:
                segment.text = f"{error.hotfix_hint}{error.word_expected}"
```

### 2. Intra-session Segment Cache

| Параметр | Значение |
|----------|---------|
| **Источник** | `GraphState.segment_audio[]` + `segment_approved[]` |
| **Индекс** | Позиционный (segment index) |
| **Поиск** | `segment_approved[i] == True` → skip |
| **Результат** | Пропуск пересинтеза approved сегментов |
| **Scope** | Только текущая сессия |

### 3. Voice ID → Speaker Mapping

| Параметр | Значение |
|----------|---------|
| **Источник** | `VOICE_MAP` в `tts_client.py` |
| **Индекс** | Статический dict (3 записи) |
| **Поиск** | `O(1)` dict lookup |
| **Результат** | `"speaker_1"` → `"中文女"` |

## Планируемый retrieval-контур (MAS-4)

### Pronunciation Memory Store

```
Key:   (word: str, voice_id: str)
Value: {
    phoneme_hint: str,      # "[ˈmɒskaʊ]"
    success_rate: float,    # 0.0-1.0
    attempts: int,
    last_used: datetime
}
Storage: Redis / SQLite
Index:   Hash index by (word, voice_id)
Search:  Exact match
Policy:  LRU eviction, max 10K entries
```

### Segment Audio Cache

```
Key:   hash(text + voice_id + emotion)
Value: {
    audio_bytes: bytes,     # WAV
    wer: float,             # Last known WER
    timestamp: datetime,
    sample_rate: int
}
Storage: Redis (binary values)
Index:   SHA-256 hash
Search:  Exact match
Policy:  TTL 24h, maxmemory 1GB
```

### Ограничения

| Ограничение | Описание |
|-------------|----------|
| Нет embedding search | Приблизительный вспоминание невозможно |
| Нет reranking | Только exact match lookup |
| Нет cross-session learning в PoC | Каждый запрос обрабатывается с чистого листа |
| Нет persistence | In-memory store теряется при рестарте |
