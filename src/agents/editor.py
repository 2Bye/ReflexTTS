"""Editor Agent — two-path audio correction.

Path 1 (Fast): Pronunciation Hotfix
  → Director adds phoneme hints → Actor re-generates → Critic re-checks
  → Handled by graph routing (already in M2)

Path 2 (Deep): Latent Inpainting via Flow Matching
  → WhisperX alignment → mel masking → FM regeneration → vocoder → blend
  → Falls back to chunk regeneration + cross-fade if FM unavailable

Flow: Critic errors → Editor → repaired audio → Critic re-validation
"""

from __future__ import annotations

from src.agents.actor import _decode_wav_to_array, _encode_wav
from src.audio.alignment import create_error_regions
from src.audio.crossfade import crossfade_chunks
from src.audio.metrics import convergence_score
from src.inference.tts_client import TTSClient
from src.log import get_logger
from src.orchestrator.state import GraphState

logger = get_logger(__name__)


async def run_editor(state: GraphState, tts: TTSClient) -> GraphState:
    """Execute the Editor Agent.

    Attempts two correction strategies:
    1. Latent inpainting (if CosyVoice3 FM module accessible)
    2. Chunk regeneration + cross-fade (fallback)

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

    # Try latent inpainting first, fall back to chunk regen
    try:
        repaired = await _inpaint_latent(state, tts)
        state.agent_log.append(
            {"agent": "editor", "action": "inpainted", "detail": "latent FM"}  # type: ignore[arg-type]
        )
    except Exception as e:
        logger.warning("editor_inpaint_fallback", error=str(e))
        repaired = await _regen_chunks(state, tts)
        state.agent_log.append(
            {"agent": "editor", "action": "chunk_regen", "detail": "crossfade fallback"}  # type: ignore[arg-type]
        )

    state.audio_bytes = repaired

    # Calculate convergence metric
    metrics = convergence_score(wer=state.wer)
    state.convergence_score = metrics.convergence_score

    logger.info(
        "editor_done",
        convergence=f"{metrics.convergence_score:.3f}",
        converged=metrics.is_converged,
    )
    return state


async def _inpaint_latent(state: GraphState, tts: TTSClient) -> bytes:
    """Attempt latent inpainting via CosyVoice3 Flow Matching.

    Pipeline:
    1. Convert error timestamps → mel regions
    2. Build binary mask with cosine taper
    3. Re-synthesize error text segments
    4. Blend original + regenerated mel-spectrograms
    5. Return repaired WAV

    Note: This is the research frontier. When CosyVoice3's internal
    FM module is not directly accessible, this falls through to the
    chunk regeneration fallback.
    """
    # Decode original audio
    original_wav = _decode_wav_to_array(state.audio_bytes)

    if len(original_wav) == 0:
        raise ValueError("Empty original audio")

    # Step 1: Create mel regions from errors
    error_dicts = [e.model_dump() for e in state.errors if not e.can_hotfix]
    regions = create_error_regions(
        error_dicts,
        sample_rate=state.sample_rate,
    )

    if not regions:
        raise ValueError("No valid error regions")

    # Step 2-5: For now, we use the chunk regen approach as the
    # CosyVoice3 FM module requires direct model access.
    # When running on GPU, this will be replaced with:
    #   mel_orig = extract_mel(original_wav, sample_rate)
    #   mask = build_inpainting_mask(regions, mel_orig.shape[1])
    #   tokens = tts._model.llm.generate(corrected_text)
    #   mel_new = tts._model.flow_matching(tokens, mask=mask)
    #   mel_blend = apply_mask_to_mel(mel_orig, mel_new, mask)
    #   wav = tts._model.vocoder(mel_blend)

    # For PoC: use chunk regen as the inpainting implementation
    raise NotImplementedError(
        "Latent FM inpainting requires direct CosyVoice3 model access. "
        "Falling back to chunk regeneration."
    )


async def _regen_chunks(state: GraphState, tts: TTSClient) -> bytes:
    """Fallback: regenerate error chunks and cross-fade into original.

    For each error region:
    1. Extract the expected text
    2. Re-synthesize that text via CosyVoice3
    3. Cross-fade the new audio into the original at the error position

    Args:
        state: Current graph state.
        tts: TTS client.

    Returns:
        Repaired WAV bytes.
    """
    original_wav = _decode_wav_to_array(state.audio_bytes)

    if len(original_wav) == 0:
        return state.audio_bytes

    result = original_wav.copy()
    sr = state.sample_rate

    for error in state.errors:
        if error.can_hotfix:
            continue

        # Convert ms → samples
        start_sample = int(error.start_ms * sr / 1000)
        end_sample = int(error.end_ms * sr / 1000)

        if start_sample >= end_sample or start_sample >= len(result):
            continue

        end_sample = min(end_sample, len(result))

        # Re-synthesize the correct word(s)
        voice_id = "speaker_1"
        if state.ssml_markup and "voice_id" in state.ssml_markup:
            voice_id = state.ssml_markup["voice_id"]

        try:
            regen_result = await tts.synthesize(
                text=error.word_expected,
                voice_id=voice_id,
            )

            result = crossfade_chunks(
                original=result,
                replacement=regen_result.waveform,
                start_sample=start_sample,
                end_sample=end_sample,
            )

            logger.debug(
                "editor_chunk_replaced",
                word=error.word_expected,
                start_ms=error.start_ms,
                end_ms=error.end_ms,
            )
        except Exception as e:
            logger.warning(
                "editor_chunk_regen_failed",
                word=error.word_expected,
                error=str(e),
            )

    return _encode_wav(result, sr)
