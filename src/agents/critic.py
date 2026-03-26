"""Critic Agent — audio quality assessment with per-segment evaluation.

Two-phase evaluation per segment:
1. ASR Phase: WhisperX transcription + forced alignment
2. Judge Phase: LLM comparison of target vs transcript per segment

Flow: audio_bytes → Critic (WhisperX + Qwen3-8B) → per-segment CriticOutput
"""

from __future__ import annotations

import json

from src.agents.actor import _decode_wav_to_array
from src.agents.pronunciation_cache import PronunciationCache
from src.agents.prompts import CRITIC_JUDGE_SYSTEM_PROMPT
from src.agents.schemas import CriticOutput
from src.inference.asr_client import ASRClient
from src.inference.vllm_client import VLLMClient
from src.log import get_logger
from src.orchestrator.state import DetectedError, GraphState

logger = get_logger(__name__)


async def run_critic(
    state: GraphState,
    asr: ASRClient,
    vllm: VLLMClient,
    pronunciation_cache: PronunciationCache | None = None,
) -> GraphState:
    """Execute the Critic Agent with per-segment evaluation.

    Evaluates each segment individually using its own audio + text:
    Phase 1: Transcribe each segment's audio with WhisperX.
    Phase 2: Compare each segment's target text to transcript using Qwen3-8B.

    This enables selective re-synthesis — only failed segments are retried.

    Args:
        state: Current graph state with audio_bytes and segment_audio.
        asr: WhisperX ASR client.
        vllm: vLLM client for the Judge LLM.
        pronunciation_cache: Optional cross-session pronunciation cache.

    Returns:
        Updated state with per-segment evaluation, errors, and approval.
    """
    logger.info("critic_start", iteration=state.iteration)

    # Extract Director segments for per-segment text
    segments: list[dict[str, object]] = []
    if state.ssml_markup and "segments" in state.ssml_markup:
        segments = state.ssml_markup["segments"]

    # ── Per-segment evaluation ──
    has_segment_audio = (
        len(state.segment_audio) == len(segments)
        and len(segments) > 0
        and all(len(sa) > 0 for sa in state.segment_audio)
    )

    if has_segment_audio and len(segments) > 1:
        # Per-segment evaluation path
        logger.info(
            "critic_per_segment_mode",
            num_segments=len(segments),
        )
        all_errors: list[DetectedError] = []
        segment_wers: list[float] = []

        # Initialize segment_approved if needed
        if len(state.segment_approved) != len(segments):
            state.segment_approved = [False] * len(segments)

        for seg_idx, seg in enumerate(segments):
            # Skip already-approved segments
            if state.segment_approved[seg_idx]:
                logger.info(
                    "critic_segment_skip_approved",
                    segment_index=seg_idx,
                )
                segment_wers.append(0.0)
                continue

            seg_text = str(seg.get("text", ""))
            seg_audio = state.segment_audio[seg_idx]

            seg_errors, seg_wer, seg_approved = await _evaluate_segment(
                seg_idx=seg_idx,
                seg_text=seg_text,
                seg_audio=seg_audio,
                sample_rate=state.sample_rate,
                asr=asr,
                vllm=vllm,
            )

            all_errors.extend(seg_errors)
            segment_wers.append(seg_wer)
            state.segment_approved[seg_idx] = seg_approved

        # Overall assessment
        state.errors = all_errors
        state.wer = sum(segment_wers) / len(segment_wers) if segment_wers else 0.0
        state.is_approved = all(state.segment_approved)

        # Build transcript from all segments
        state.transcript = "(per-segment evaluation)"

        approved_count = sum(1 for a in state.segment_approved if a)
        state.agent_log.append(
            {  # type: ignore[arg-type]
                "agent": "critic",
                "action": "evaluated",
                "detail": (
                    f"Per-segment: {approved_count}/{len(segments)} approved, "
                    f"WER={state.wer:.3f}, errors={len(all_errors)}"
                ),
            }
        )

        logger.info(
            "critic_done_per_segment",
            wer=f"{state.wer:.3f}",
            errors=len(all_errors),
            is_approved=state.is_approved,
            segments_approved=f"{approved_count}/{len(segments)}",
        )

        # Record pronunciation cache results
        if pronunciation_cache and state.errors:
            await _record_pronunciation_results(
                state=state,
                pronunciation_cache=pronunciation_cache,
            )

    else:
        # Fallback: whole-audio evaluation (single segment or no segment_audio)
        await _evaluate_whole_audio(state, asr, vllm)

    return state


async def _evaluate_segment(
    *,
    seg_idx: int,
    seg_text: str,
    seg_audio: bytes,
    sample_rate: int,
    asr: ASRClient,
    vllm: VLLMClient,
) -> tuple[list[DetectedError], float, bool]:
    """Evaluate a single segment's audio against its target text.

    Returns:
        Tuple of (errors, wer, is_approved) for this segment.
    """
    # Phase 1: ASR on segment audio
    audio_array = _decode_wav_to_array(seg_audio)

    if len(audio_array) == 0:
        logger.warning("critic_segment_empty_audio", segment_index=seg_idx)
        return [], 0.0, True

    logger.info(
        "critic_segment_asr_start",
        segment_index=seg_idx,
        audio_samples=len(audio_array),
        target_text=seg_text[:100],
    )

    asr_result = await asr.transcribe(
        audio=audio_array,
        sample_rate=sample_rate,
    )

    logger.info(
        "critic_segment_asr_done",
        segment_index=seg_idx,
        transcript=asr_result.text,
        words=len(asr_result.word_timestamps),
    )

    # Phase 2: Judge (LLM comparison) — full text, no truncation
    word_ts = [
        {
            "word": w.word,
            "start_ms": w.start_ms,
            "end_ms": w.end_ms,
            "score": w.score,
        }
        for w in asr_result.word_timestamps
    ]

    judge_input = json.dumps(
        {
            "target_text": seg_text,
            "transcript": asr_result.text,
            "word_timestamps": word_ts[:20],
            "segment_index": seg_idx,
        },
        ensure_ascii=False,
    )

    critic_output = await vllm.chat_json(
        system_prompt=CRITIC_JUDGE_SYSTEM_PROMPT,
        user_message=judge_input,
        response_model=CriticOutput,
        max_tokens=4096,
    )

    # Convert CriticError → DetectedError with segment_index
    errors: list[DetectedError] = []
    for err in critic_output.errors:
        errors.append(
            DetectedError(
                word_expected=err.word_expected,
                word_actual=err.word_actual,
                start_ms=err.start_ms,
                end_ms=err.end_ms,
                severity=err.severity.value,
                description=err.description,
                can_hotfix=err.can_hotfix,
                hotfix_hint=err.hotfix_hint,
                segment_index=seg_idx,
            )
        )

    logger.info(
        "critic_segment_judge_result",
        segment_index=seg_idx,
        is_approved=critic_output.is_approved,
        wer=f"{critic_output.wer:.3f}",
        errors_count=len(errors),
    )

    for detected in errors:
        logger.info(
            "critic_segment_error",
            segment_index=seg_idx,
            expected=detected.word_expected,
            actual=detected.word_actual,
            severity=detected.severity,
        )

    return errors, critic_output.wer, critic_output.is_approved


async def _evaluate_whole_audio(
    state: GraphState,
    asr: ASRClient,
    vllm: VLLMClient,
) -> None:
    """Fallback: evaluate the whole audio at once (original behavior).

    Used when there's only one segment or no per-segment audio available.
    """
    logger.info("critic_whole_audio_mode")

    # Phase 1: ASR
    audio_array = _decode_wav_to_array(state.audio_bytes)
    logger.info(
        "critic_asr_input",
        audio_samples=len(audio_array),
        sample_rate=state.sample_rate,
        audio_duration_s=f"{len(audio_array) / state.sample_rate:.2f}",
    )

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
        transcript=asr_result.text,
        transcript_length=len(asr_result.text),
        words=len(asr_result.word_timestamps),
        language=asr_result.language,
    )

    # Phase 2: Judge — full text for single segment, no truncation needed
    target_text = _extract_target_text(state)

    judge_input = json.dumps(
        {
            "target_text": target_text,
            "transcript": asr_result.text,
            "word_timestamps": state.word_timestamps[:20],
        },
        ensure_ascii=False,
    )

    critic_output = await vllm.chat_json(
        system_prompt=CRITIC_JUDGE_SYSTEM_PROMPT,
        user_message=judge_input,
        response_model=CriticOutput,
        max_tokens=4096,
    )

    # Update state — set segment_index=0 for single-segment
    state.errors = [
        DetectedError(
            word_expected=e.word_expected,
            word_actual=e.word_actual,
            start_ms=e.start_ms,
            end_ms=e.end_ms,
            severity=e.severity.value,
            description=e.description,
            can_hotfix=e.can_hotfix,
            hotfix_hint=e.hotfix_hint,
            segment_index=getattr(e, "segment_index", 0),
        )
        for e in critic_output.errors
    ]
    state.wer = critic_output.wer
    state.is_approved = critic_output.is_approved

    # Update segment_approved for single segment
    if state.segment_approved:
        state.segment_approved = [critic_output.is_approved] * len(state.segment_approved)

    logger.info(
        "critic_judge_result",
        is_approved=critic_output.is_approved,
        wer=f"{critic_output.wer:.3f}",
        errors_count=len(critic_output.errors),
        summary=critic_output.summary,
    )

    for err in critic_output.errors:
        logger.info(
            "critic_error",
            expected=err.word_expected,
            actual=err.word_actual,
            severity=err.severity,
            can_hotfix=err.can_hotfix,
        )

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
    )


def _extract_target_text(state: GraphState) -> str:
    """Extract the original target text from Director output or raw input."""
    if state.ssml_markup and "segments" in state.ssml_markup:
        segments = state.ssml_markup["segments"]
        return " ".join(seg.get("text", "") for seg in segments)
    return state.text


async def _record_pronunciation_results(
    *,
    state: GraphState,
    pronunciation_cache: PronunciationCache,
) -> None:
    """Record pronunciation hint results into the cross-session cache.

    For errors with hotfix hints: if the segment is now approved after
    using the hint, record success. Otherwise record failure.
    """
    for error in state.errors:
        if not error.hotfix_hint:
            continue

        seg_idx = error.segment_index
        is_approved = (
            seg_idx >= 0
            and seg_idx < len(state.segment_approved)
            and state.segment_approved[seg_idx]
        )

        await pronunciation_cache.record(
            word=error.word_expected,
            voice_id=state.voice_id,
            hint=error.hotfix_hint,
            success=is_approved,
        )
