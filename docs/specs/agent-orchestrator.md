# Spec: Agent / Orchestrator

> Steps, transition rules, stop conditions, retry/fallback.

---

## 1. LangGraph Orchestrator

**Files:** `src/orchestrator/graph.py`, `src/orchestrator/state.py`

### Graph Structure

```
Nodes: director, actor, critic, editor, mark_human_review
Edges:
  director → actor (unconditional)
  actor → critic (unconditional)
  critic → {approved, hotfix, editor, needs_human_review} (conditional)
  editor → critic (unconditional)
  mark_human_review → END (unconditional)
  approved → END
  hotfix → director
```

### Node Lifecycle

Each node:
1. Receives `state: dict[str, Any]`
2. Deserializes: `gs = GraphState.model_validate(state)`
3. Executes business logic (async)
4. Serializes: `return gs.model_dump()`
5. LangGraph updates state and transitions to the next node

---

## 2. Pipeline Steps

### Step 1: Director

| Parameter | Value |
|-----------|-------|
| **Input** | `state.text`, `state.voice_id`, `state.errors` (if iteration > 0) |
| **Action** | `vllm.chat_json(DIRECTOR_PROMPT, text)` → `DirectorOutput` |
| **Output** | `state.ssml_markup`, `state.tts_instruct` |
| **Side action** | `_apply_hotfix_hints()` if iteration > 0; `_apply_cached_hints()` from `PronunciationCache` |
| **Duration** | 2-15s |

### Step 2: Actor

| Parameter | Value |
|-----------|-------|
| **Input** | `state.ssml_markup.segments[]`, `state.segment_approved[]` |
| **Action** | ∀ unapproved segment: check `SegmentCache` → `tts.synthesize()` via `asyncio.gather(Semaphore(4))` |
| **Output** | `state.audio_bytes`, `state.segment_audio[]`, `state.sample_rate` |
| **Optimization** | Skip approved segments; skip cached segments (`SegmentCache` hit) |
| **Duration** | 3-15s (depends on segment count) |

### Step 3: Critic

| Parameter | Value |
|-----------|-------|
| **Input** | `state.segment_audio[]`, `state.ssml_markup.segments[]` |
| **Action 1** | ∀ unapproved segment: `asr.transcribe(segment_wav)` |
| **Action 2** | ∀ unapproved segment: `vllm.chat_json(JUDGE_PROMPT, {target, transcript})` |
| **Output** | `state.errors[]`, `state.wer`, `state.is_approved`, `state.segment_approved[]` |
| **Post-action** | `state.iteration += 1`; record results to `PronunciationCache` |
| **Duration** | 3-15s per unapproved segment |

### Step 4: Editor (conditional)

| Parameter | Value |
|-----------|-------|
| **Input** | `state.errors[]`, `state.segment_approved[]`, `state.segment_audio[]` |
| **Action** | ∀ failed segment: `tts.synthesize(full_segment_text)` |
| **Output** | Updated `state.segment_audio[]`, rebuilt `state.audio_bytes`, `state.convergence_score` |
| **Skip conditions** | No errors; all errors can_hotfix |
| **Duration** | 1-8s per failed segment |

### Step 5: Mark Human Review (conditional)

| Parameter | Value |
|-----------|-------|
| **Input** | `state` (after max retries) |
| **Action** | Set `state.needs_human_review = True` |
| **Output** | → END |

---

## 3. Routing Rules

### `route_after_critic(state) → str`

```python
def route_after_critic(state):
    gs = GraphState.model_validate(state)

    # 1. Approved → done
    if gs.is_approved:
        return "approved"

    # 2. Max retries → escalate
    if gs.iteration >= max_retries:
        return "needs_human_review"

    # 3. Filter to unapproved segment errors
    unapproved_errors = [
        e for e in gs.errors
        if e.segment_index < 0 or (
            e.segment_index < len(gs.segment_approved)
            and not gs.segment_approved[e.segment_index]
        )
    ]

    # 4. No errors but not approved → safety fallback
    if not unapproved_errors:
        return "hotfix"

    # 5. All hotfixable → Director
    if all(e.can_hotfix for e in unapproved_errors):
        return "hotfix"

    # 6. Non-hotfix errors → Editor
    return "editor"
```

### Transition Table

| From | To | Condition |
|------|----|-----------|
| `[*]` | `director` | Entry point |
| `director` | `actor` | Always |
| `actor` | `critic` | Always |
| `critic` | END | `is_approved == True` |
| `critic` | `mark_human_review` | `iteration >= max_retries` |
| `critic` | `director` | All unapproved errors are `can_hotfix` |
| `critic` | `editor` | Non-hotfix errors exist |
| `editor` | `critic` | Always (re-evaluate) |
| `mark_human_review` | END | Always |

---

## 4. Stop Conditions

| Condition | Trigger | Result |
|-----------|---------|--------|
| **All segments approved** | `is_approved == True` | ✅ Final audio returned |
| **Max retries reached** | `iteration >= max_retries` (default 5) | ⚠️ Best-effort audio + `needs_human_review` |
| **Pipeline timeout** | `asyncio.wait_for(300s)` | ❌ `status=failed`, `error_message=timeout` |
| **Exception** | Any unhandled exception | ❌ `status=failed`, logged |

---

## 5. Retry / Fallback Strategy

### Retry Levels

| Level | Mechanism | Max attempts | Backoff |
|-------|----------|-------------|---------|
| **L1: vLLM connection** | `_request_with_retry()` | 5 | Exponential: 2^n sec |
| **L2: JSON parsing** | strip → parse → brace extract | 3 steps | No wait |
| **L3: Pipeline iteration** | Critic → Director/Editor loop | max_retries (5) | No wait |
| **L4: Pipeline timeout** | `asyncio.wait_for()` | 1 (hard timeout) | — |

### Fallback Chain

```
1. JSON parse fail → strip <think> → brace extraction → VLLMResponseError
2. Pronunciation error, can_hotfix → Director (phoneme hints)
3. Pronunciation error, !can_hotfix → Editor (full re-synth)
4. Persistent error after max_retries → Human review (needs_human_review=True)
5. Pipeline timeout 300s → Session failed
6. Unknown emotion → Mapped to "neutral" (no crash)
```

---

## 6. Per-Agent Error Handling

| Agent | Error | Action |
|-------|-------|--------|
| Director | vLLM timeout | 5× retry → pipeline fail |
| Director | Invalid JSON | Brace extraction fallback |
| Director | Unknown emotion | Auto-map to "neutral" |
| Actor | TTS error | Pipeline fail |
| Actor | Approved segment | Skip (use cached audio) |
| Critic | ASR error | Pipeline fail |
| Critic | Judge JSON error | Brace extraction, then pipeline fail |
| Editor | No failed segments | Skip (is_approved=True) |
| Editor | All errors can_hotfix | Skip (→ Director hotfix) |
| Editor | TTS re-synth error | Pipeline fail |
