"""Editor Agent — segment-level audio correction.

Path 1 (Fast): Pronunciation Hotfix
  → Director adds phoneme hints → Actor re-generates → Critic re-checks
  → Handled by graph routing (already in M2)

Path 2 (Deep): Segment Re-synthesis
  → Re-synthesize entire failed segments via CosyVoice3 (clean TTS, no crossfade)
  → Rebuild combined audio from per-segment WAV data

Path 3 (Future): Latent Inpainting via Flow Matching
  → WhisperX alignment → mel masking → FM regeneration → vocoder → blend

Flow: Critic errors → Editor → repaired audio → Critic re-validation
"""

from __future__ import annotations

import numpy as np

from src.agents.actor import _decode_wav_to_array, _encode_wav
from src.audio.metrics import convergence_score
from src.inference.tts_client import TTSClient
from src.log import get_logger
from src.orchestrator.state import GraphState

logger = get_logger(__name__)


async def run_editor(state: GraphState, tts: TTSClient) -> GraphState:
    """Execute the Editor Agent with segment-level re-synthesis.

    New strategy (M11): instead of patching individual words via crossfade,
    re-synthesize entire failed segments cleanly. This produces artifact-free
    audio by replacing whole segment WAVs.

    Args:
        state: Current graph state with errors and audio.
        tts: CosyVoice3 TTS client.

    Returns:
        Updated state with repaired audio.
    """
    if not state.errors:
        logger.info("editor_skip_no_errors")
        state.is_approved = True
        return state

    non_hotfix_errors = [
        e for e in state.errors
        if not e.can_hotfix
    ]

    if not non_hotfix_errors:
        logger.info("editor_skip_all_hotfixable")
        return state

    logger.info(
        "editor_start",
        error_count=len(non_hotfix_errors),
        iteration=state.iteration,
    )

    # Identify which segments failed
    failed_segments = _get_failed_segments(state)

    if failed_segments and state.segment_audio:
        # M11: Segment-level re-synthesis (clean, no crossfade)
        await _regen_segments(state, tts, failed_segments)
        state.agent_log.append(
            {  # type: ignore[arg-type]
                "agent": "editor",
                "action": "segment_regen",
                "detail": f"re-synthesized segments {failed_segments}",
            }
        )
    else:
        # Legacy fallback: no segment data available
        logger.warning("editor_no_segment_data_fallback")
        state.agent_log.append(
            {  # type: ignore[arg-type]
                "agent": "editor",
                "action": "skip",
                "detail": "no segment data for targeted regen",
            }
        )

    # Calculate convergence metric
    metrics = convergence_score(wer=state.wer)
    state.convergence_score = metrics.convergence_score

    logger.info(
        "editor_done",
        convergence=f"{metrics.convergence_score:.3f}",
        converged=metrics.is_converged,
        failed_segments=failed_segments,
    )
    return state


def _get_failed_segments(state: GraphState) -> list[int]:
    """Identify which segments failed based on errors and segment_approved.

    Returns:
        Sorted list of segment indices that need re-synthesis.
    """
    failed: set[int] = set()

    # From segment_approved flags
    for i, approved in enumerate(state.segment_approved):
        if not approved:
            failed.add(i)

    # From error segment_index
    for err in state.errors:
        if not err.can_hotfix and err.segment_index >= 0:
            failed.add(err.segment_index)

    return sorted(failed)


async def _regen_segments(
    state: GraphState,
    tts: TTSClient,
    failed_segments: list[int],
) -> None:
    """Re-synthesize entire failed segments and rebuild combined audio.

    Unlike the old crossfade approach, this produces clean full-segment
    audio with no splicing artifacts.

    Args:
        state: Current graph state (modified in-place).
        tts: CosyVoice3 TTS client.
        failed_segments: Indices of segments to re-synthesize.
    """
    # Get Director segments for text
    segments: list[dict[str, object]] = []
    voice_id = "speaker_1"
    if state.ssml_markup:
        if "segments" in state.ssml_markup:
            segments = state.ssml_markup["segments"]
        if "voice_id" in state.ssml_markup:
            voice_id = str(state.ssml_markup["voice_id"]) or voice_id


    for seg_idx in failed_segments:
        if seg_idx >= len(segments):
            logger.warning(
                "editor_segment_out_of_range",
                segment_index=seg_idx,
                total_segments=len(segments),
            )
            continue

        seg = segments[seg_idx]
        seg_text = str(seg.get("text", ""))

        if not seg_text:
            continue

        # Build instruct from emotion
        instruct = ""
        emotion = str(seg.get("emotion", "neutral"))
        if emotion != "neutral":
            instruct = f"Speak with {emotion} tone and feeling."

        logger.info(
            "editor_regen_segment",
            segment_index=seg_idx,
            text=seg_text[:80],
            emotion=emotion,
        )

        try:
            result = await tts.synthesize(
                text=seg_text,
                voice_id=voice_id,
                instruct=instruct,
            )

            # Replace segment audio
            state.segment_audio[seg_idx] = _encode_wav(
                result.waveform, result.sample_rate
            )
            # Reset approval — Critic will re-evaluate
            state.segment_approved[seg_idx] = False

            logger.info(
                "editor_segment_done",
                segment_index=seg_idx,
                duration_s=f"{result.duration_seconds:.2f}",
                samples=len(result.waveform),
            )
        except Exception as e:
            logger.warning(
                "editor_segment_regen_failed",
                segment_index=seg_idx,
                error=str(e),
            )

    # Rebuild combined audio from all segments (approved + re-synthesized)
    _rebuild_combined_audio(state, segments)


def _rebuild_combined_audio(
    state: GraphState,
    segments: list[dict[str, object]],
) -> None:
    """Rebuild the combined audio from per-segment WAV data.

    Concatenates all segment audio (with pauses) into the final audio_bytes.

    Args:
        state: Current graph state (modified in-place).
        segments: Director segments (for pause info).
    """
    sr = state.sample_rate
    waveforms: list[np.ndarray] = []

    for i, seg in enumerate(segments):
        # Insert pause before segment
        pause_val = seg.get("pause_before_ms", 0)
        pause_ms = int(float(str(pause_val))) if pause_val else 0
        if pause_ms > 0:
            pause_samples = int(sr * pause_ms / 1000)
            waveforms.append(np.zeros(pause_samples, dtype=np.float32))

        # Decode segment audio
        if i < len(state.segment_audio) and state.segment_audio[i]:
            seg_wav = _decode_wav_to_array(state.segment_audio[i])
            if len(seg_wav) > 0:
                waveforms.append(seg_wav)

    if waveforms:
        combined = np.concatenate(waveforms)
        state.audio_bytes = _encode_wav(combined, sr)
    else:
        logger.warning("editor_rebuild_empty")

    logger.info(
        "editor_rebuild_done",
        segments=len(segments),
        total_samples=sum(len(w) for w in waveforms),
    )
