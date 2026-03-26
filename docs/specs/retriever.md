# Spec: Retriever / Retrieval

> Data sources, indexes, search, reranking, limitations.

---

## Current State: Retrieval-free

The PoC version of ReflexTTS has **no classical retrieval pipeline** (vector store, embedding search, RAG). The system operates in generative-only mode.

## Retrieval-like Mechanisms (Implemented)

### 1. Intra-session Phoneme Lookup

| Parameter | Value |
|-----------|-------|
| **Source** | `GraphState.errors[].hotfix_hint` (from Critic) |
| **Index** | None — direct iteration over `errors[]` |
| **Search** | `word_expected in segment.text` (substring match) |
| **Result** | Phoneme hint injection into segment text |
| **Scope** | Current session only (intra-session) |

```python
# director.py → _apply_hotfix_hints()
for error in state.errors:
    if error.can_hotfix and error.hotfix_hint:
        for segment in output.segments:
            if error.word_expected in segment.text:
                segment.text = f"{error.hotfix_hint}{error.word_expected}"
```

### 2. Cross-session Pronunciation Cache

| Parameter | Value |
|-----------|-------|
| **Source** | `PronunciationCache` (word + voice_id → phoneme hint) |
| **Index** | In-memory dict, key = `(word.lower(), voice_id)` |
| **Search** | Exact match, threshold = 2 successful uses |
| **Result** | Proactive phoneme hint injection before synthesis |
| **Scope** | ✅ Cross-session (persistent across requests) |

### 3. Cross-session Segment Audio Cache

| Parameter | Value |
|-----------|-------|
| **Source** | `SegmentCache` (text + voice + emotion → WAV bytes) |
| **Index** | SHA-256 hash of `(text, voice_id, emotion)` |
| **Search** | Exact match, WER=0 required for caching |
| **Result** | Skip GPU synthesis entirely for cached segments |
| **Scope** | ✅ Cross-session, TTL 24h, LRU eviction |

### 4. Intra-session Segment Cache

| Parameter | Value |
|-----------|-------|
| **Source** | `GraphState.segment_audio[]` + `segment_approved[]` |
| **Index** | Positional (segment index) |
| **Search** | `segment_approved[i] == True` → skip |
| **Result** | Skip re-synthesis of approved segments |
| **Scope** | Current session only |

### 5. Voice ID → Speaker Mapping

| Parameter | Value |
|-----------|-------|
| **Source** | `VOICE_MAP` in `tts_client.py` |
| **Index** | Static dict (3 entries) |
| **Search** | `O(1)` dict lookup |
| **Result** | `"speaker_1"` → `"中文女"` |

## Future Retrieval (MAS-4)

### Pronunciation Memory Store (Redis backend)

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

### Segment Audio Cache (Redis backend)

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

### Limitations

| Limitation | Description |
|------------|-------------|
| No embedding search | Approximate recall not possible |
| No reranking | Exact match lookup only |
| No persistence (PoC) | In-memory caches lost on restart |
