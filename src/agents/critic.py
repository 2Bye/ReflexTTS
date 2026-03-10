"""Critic Agent — audio quality assessment.

Two-phase evaluation:
1. ASR Phase: WhisperX transcription + forced alignment
2. Judge Phase: LLM comparison of target vs transcript

Flow: audio_bytes → Critic (WhisperX + Qwen3-8B) → CriticOutput
"""

from __future__ import annotations

import json

from src.agents.actor import _decode_wav_to_array
from src.agents.prompts import CRITIC_JUDGE_SYSTEM_PROMPT
from src.agents.schemas import CriticOutput
from src.inference.asr_client import ASRClient
from src.inference.vllm_client import VLLMClient
from src.log import get_logger
from src.orchestrator.state import GraphState

logger = get_logger(__name__)


async def run_critic(
    state: GraphState,
    asr: ASRClient,
    vllm: VLLMClient,
) -> GraphState:
    """Execute the Critic Agent (ASR + Judge).

    Phase 1: Transcribe the generated audio with WhisperX.
    Phase 2: Compare target text to transcript using Qwen3-8B.

    Args:
        state: Current graph state with audio_bytes.
        asr: WhisperX ASR client.
        vllm: vLLM client for the Judge LLM.

    Returns:
        Updated state with transcript, errors, and approval status.
    """
    logger.info("critic_start", iteration=state.iteration)

    # ── Phase 1: ASR Transcription ──
    audio_array = _decode_wav_to_array(state.audio_bytes)

    asr_result = await asr.transcribe(
        audio=audio_array,
        sample_rate=state.sample_rate,
    )

    state.transcript = asr_result.text
    state.word_timestamps = [
        {
            "word": w.word,
            "start_ms": w.start_ms,
            "end_ms": w.end_ms,
            "score": w.score,
        }
        for w in asr_result.word_timestamps
    ]

    logger.info(
        "critic_asr_done",
        transcript_length=len(asr_result.text),
        words=len(asr_result.word_timestamps),
    )

    # ── Phase 2: Judge (LLM comparison) ──
    target_text = _extract_target_text(state)

    judge_input = json.dumps(
        {
            "target_text": target_text,
            "transcript": asr_result.text,
            "word_timestamps": state.word_timestamps[:50],
        },
        ensure_ascii=False,
    )

    critic_output = await vllm.chat_json(
        system_prompt=CRITIC_JUDGE_SYSTEM_PROMPT,
        user_message=judge_input,
        response_model=CriticOutput,
    )

    # Update state
    state.errors = critic_output.errors  # type: ignore[assignment]
    state.wer = critic_output.wer
    state.is_approved = critic_output.is_approved

    state.agent_log.append(
        {  # type: ignore[arg-type]
            "agent": "critic",
            "action": "evaluated",
            "detail": (
                f"WER={critic_output.wer:.3f}, errors={len(critic_output.errors)}, "
                f"approved={critic_output.is_approved}"
            ),
        }
    )

    logger.info(
        "critic_done",
        wer=f"{critic_output.wer:.3f}",
        errors=len(critic_output.errors),
        is_approved=critic_output.is_approved,
        summary=critic_output.summary[:100],
    )
    return state


def _extract_target_text(state: GraphState) -> str:
    """Extract the original target text from Director output or raw input."""
    if state.ssml_markup and "segments" in state.ssml_markup:
        segments = state.ssml_markup["segments"]
        return " ".join(seg.get("text", "") for seg in segments)
    return state.text
