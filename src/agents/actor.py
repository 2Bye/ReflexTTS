"""Actor Agent — speech synthesis.

Takes Director output and generates audio via CosyVoice3.
Handles segment concatenation and pause insertion.

Flow: DirectorOutput → Actor (CosyVoice3) → audio_bytes
"""

from __future__ import annotations

import struct
from io import BytesIO

import numpy as np

from src.agents.schemas import DirectorOutput, Segment
from src.inference.tts_client import TTSClient
from src.log import get_logger
from src.orchestrator.state import GraphState

logger = get_logger(__name__)


async def run_actor(state: GraphState, tts: TTSClient) -> GraphState:
    """Execute the Actor Agent.

    Synthesizes speech from Director's segments and produces audio.
    On retry iterations, only re-synthesizes unapproved segments,
    reusing cached audio for segments that passed Critic evaluation.

    Args:
        state: Current graph state with DirectorOutput in ssml_markup.
        tts: CosyVoice3 TTS client.

    Returns:
        Updated state with audio_bytes and segment_audio.
    """
    director_output = DirectorOutput.model_validate(state.ssml_markup)
    voice_id = director_output.voice_id or state.voice_id
    num_segments = len(director_output.segments)

    logger.info(
        "actor_start",
        segments=num_segments,
        voice=voice_id,
        iteration=state.iteration,
    )

    # Initialize segment_audio list if needed
    if len(state.segment_audio) != num_segments:
        state.segment_audio = [b""] * num_segments
        state.segment_approved = [False] * num_segments

    waveforms: list[np.ndarray] = []
    sample_rate = tts.sample_rate

    for i, segment in enumerate(director_output.segments):
        # Skip re-synthesis for already-approved segments
        if state.segment_approved[i] and state.segment_audio[i]:
            logger.info(
                "actor_segment_cached",
                segment_index=i,
                text=segment.text[:50],
            )
            seg_wav = _decode_wav_to_array(state.segment_audio[i])
            waveforms.append(seg_wav)
            continue

        # Insert pause before segment
        if segment.pause_before_ms > 0:
            pause_samples = int(sample_rate * segment.pause_before_ms / 1000)
            waveforms.append(np.zeros(pause_samples, dtype=np.float32))

        # Build text with phoneme hints inline
        text = _build_text_with_hints(segment)

        # Build instruct from emotion
        instruct = ""
        if segment.emotion.value != "neutral":
            instruct = f"Speak with {segment.emotion.value} tone and feeling."

        logger.info(
            "actor_segment_start",
            segment_index=i,
            text=text,
            emotion=segment.emotion.value,
            instruct=instruct,
        )

        # Synthesize this segment
        result = await tts.synthesize(
            text=text,
            voice_id=voice_id,
            instruct=instruct,
        )

        waveforms.append(result.waveform)
        sample_rate = result.sample_rate

        # Store per-segment audio
        state.segment_audio[i] = _encode_wav(result.waveform, sample_rate)

        logger.info(
            "actor_segment_done",
            segment_index=i,
            duration_s=f"{result.duration_seconds:.2f}",
            waveform_samples=len(result.waveform),
            sample_rate=result.sample_rate,
        )

    # Concatenate all segments
    combined = np.concatenate(waveforms) if waveforms else np.array([], dtype=np.float32)

    # Encode to WAV bytes
    wav_bytes = _encode_wav(combined, sample_rate)

    state.audio_bytes = wav_bytes
    state.sample_rate = sample_rate

    # Count how many segments were re-synthesized vs cached
    cached_count = sum(1 for a in state.segment_approved if a)
    synth_count = num_segments - cached_count

    state.agent_log.append(
        {  # type: ignore[arg-type]
            "agent": "actor",
            "action": "synthesized",
            "detail": f"{len(combined) / sample_rate:.2f}s audio ({synth_count} new, {cached_count} cached)",
        }
    )

    logger.info(
        "actor_done",
        total_duration_s=f"{len(combined) / sample_rate:.2f}",
        wav_size_kb=len(wav_bytes) // 1024,
        total_segments=num_segments,
        segments_synthesized=synth_count,
        segments_cached=cached_count,
    )
    return state


def _build_text_with_hints(segment: Segment) -> str:
    """Apply phoneme hints to segment text for CosyVoice3 inline correction."""
    text = segment.text
    # Phoneme hints are already embedded in the text by Director hotfix
    # No additional processing needed — CosyVoice3 reads inline pinyin/CMU
    return text


def _encode_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 waveform to WAV bytes (16-bit PCM).

    Args:
        waveform: Audio waveform as float32 numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        WAV file as bytes.
    """
    # Normalize to int16 range
    audio_int16 = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    data = audio_int16.tobytes()

    buf = BytesIO()
    num_channels = 1
    sample_width = 2  # 16-bit

    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
    buf.write(struct.pack("<H", num_channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)

    return buf.getvalue()


def _decode_wav_to_array(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 numpy array.

    Shared utility used by Critic and Editor agents.

    Args:
        wav_bytes: WAV file bytes (16-bit PCM).

    Returns:
        Audio waveform as float32 array, normalized to [-1, 1].
    """
    if len(wav_bytes) < 44:
        return np.array([], dtype=np.float32)

    data_offset = wav_bytes.find(b"data")
    if data_offset == -1:
        return np.array([], dtype=np.float32)

    data_size = struct.unpack("<I", wav_bytes[data_offset + 4 : data_offset + 8])[0]
    audio_data = wav_bytes[data_offset + 8 : data_offset + 8 + data_size]

    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32767.0
