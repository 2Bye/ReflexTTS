"""Director Agent — text analysis and speech preparation.

Takes raw input text and produces structured synthesis instructions:
segments, emotions, pauses, and phoneme hints.

Flow: User text → Director (Qwen3-8B) → DirectorOutput → Actor
"""

from __future__ import annotations

from src.agents.pronunciation_cache import PronunciationCache

from src.agents.prompts import DIRECTOR_SYSTEM_PROMPT
from src.agents.schemas import DirectorOutput
from src.inference.vllm_client import VLLMClient
from src.log import get_logger
from src.orchestrator.state import GraphState

logger = get_logger(__name__)


async def run_director(
    state: GraphState,
    vllm: VLLMClient,
    pronunciation_cache: PronunciationCache | None = None,
) -> GraphState:
    """Execute the Director Agent.

    Analyzes input text and produces structured synthesis instructions.

    Args:
        state: Current graph state with input text.
        vllm: vLLM client for LLM inference.
        pronunciation_cache: Optional cross-session pronunciation cache.

    Returns:
        Updated state with DirectorOutput in ssml_markup.
    """
    logger.info("director_start", text_length=len(state.text), voice=state.voice_id)
    logger.debug("director_input", text_length=len(state.text))

    director_output = await vllm.chat_json(
        system_prompt=DIRECTOR_SYSTEM_PROMPT,
        user_message=state.text,
        response_model=DirectorOutput,
    )

    logger.info(
        "director_llm_response",
        segments=len(director_output.segments),
        voice=director_output.voice_id,
        language=director_output.language,
        notes=director_output.notes,
    )
    for i, seg in enumerate(director_output.segments):
        logger.info(
            "director_segment",
            index=i,
            text=seg.text,
            emotion=seg.emotion.value,
            pause_before_ms=seg.pause_before_ms,
            phoneme_hints=seg.phoneme_hints,
        )

    # Override voice_id if user specified one
    if state.voice_id:
        director_output.voice_id = state.voice_id

    # Apply any phoneme hints from previous hotfix iterations
    if state.iteration > 0 and state.errors:
        director_output = _apply_hotfix_hints(director_output, state)

    # Apply cached pronunciation hints (cross-session learning)
    if pronunciation_cache:
        cached_hints = await pronunciation_cache.get_hints_for_text(
            state.text, state.voice_id
        )
        if cached_hints:
            director_output = _apply_cached_hints(director_output, cached_hints)
            logger.info(
                "director_cached_hints_applied",
                hints_count=len(cached_hints),
                words=list(cached_hints.keys()),
            )

    # Build instruct string for TTS from the first segment's emotion
    instruct_parts: list[str] = []
    for seg in director_output.segments:
        if seg.emotion.value != "neutral":
            instruct_parts.append(f"Speak with {seg.emotion.value} tone.")
        if seg.phoneme_hints:
            instruct_parts.append(f"Pronunciation hints: {', '.join(seg.phoneme_hints)}")

    state.ssml_markup = director_output.model_dump()
    state.tts_instruct = " ".join(instruct_parts)
    state.agent_log.append(
        {"agent": "director", "action": "analyzed", "detail": f"{len(director_output.segments)} segments"}  # type: ignore[arg-type]
    )

    logger.info(
        "director_done",
        segments=len(director_output.segments),
        instruct=state.tts_instruct[:200],
    )
    return state


def _apply_hotfix_hints(output: DirectorOutput, state: GraphState) -> DirectorOutput:
    """Inject phoneme hotfix hints from Critic errors into Director output.

    When the Critic found pronunciation errors that can be fixed with
    inline phoneme hints, this function injects those hints back
    into the Director segments for retry.
    """
    for error in state.errors:
        if not error.can_hotfix or not error.hotfix_hint:
            continue

        # Find the segment containing the error word and add the hint
        for segment in output.segments:
            if error.word_expected in segment.text:
                segment.phoneme_hints.append(error.hotfix_hint)
                # Replace the word with the phoneme-annotated version
                segment.text = segment.text.replace(
                    error.word_expected,
                    f"{error.hotfix_hint}{error.word_expected}",
                    1,
                )
                logger.debug(
                    "hotfix_applied",
                    word=error.word_expected,
                    hint=error.hotfix_hint,
                )
                break

    return output


def _apply_cached_hints(output: DirectorOutput, hints: dict[str, str]) -> DirectorOutput:
    """Apply cached pronunciation hints from cross-session memory.

    Proactively injects known-good phoneme hints for difficult words
    before synthesis, avoiding unnecessary correction iterations.
    """
    for word, hint in hints.items():
        for segment in output.segments:
            if word in segment.text.lower():
                # Find the original-case word in the segment
                import re

                match = re.search(re.escape(word), segment.text, re.IGNORECASE)
                if match:
                    original_word = match.group()
                    segment.text = segment.text.replace(
                        original_word,
                        f"{hint}{original_word}",
                        1,
                    )
                    segment.phoneme_hints.append(hint)
                    logger.debug(
                        "cached_hint_applied",
                        word=original_word,
                        hint=hint,
                    )
                    break
    return output
