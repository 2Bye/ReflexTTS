"""Director Agent — text analysis and speech preparation.

Takes raw input text and produces structured synthesis instructions:
segments, emotions, pauses, and phoneme hints.

Flow: User text → Director (Qwen3-8B) → DirectorOutput → Actor
"""

from __future__ import annotations

from src.agents.prompts import DIRECTOR_SYSTEM_PROMPT
from src.agents.schemas import DirectorOutput
from src.inference.vllm_client import VLLMClient
from src.log import get_logger
from src.orchestrator.state import GraphState

logger = get_logger(__name__)


async def run_director(state: GraphState, vllm: VLLMClient) -> GraphState:
    """Execute the Director Agent.

    Analyzes input text and produces structured synthesis instructions.

    Args:
        state: Current graph state with input text.
        vllm: vLLM client for LLM inference.

    Returns:
        Updated state with DirectorOutput in ssml_markup.
    """
    logger.info("director_start", text_length=len(state.text), voice=state.voice_id)

    director_output = await vllm.chat_json(
        system_prompt=DIRECTOR_SYSTEM_PROMPT,
        user_message=state.text,
        response_model=DirectorOutput,
    )

    # Override voice_id if user specified one
    if state.voice_id:
        director_output.voice_id = state.voice_id

    # Apply any phoneme hints from previous hotfix iterations
    if state.iteration > 0 and state.errors:
        director_output = _apply_hotfix_hints(director_output, state)

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
        instruct=state.tts_instruct[:100],
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
