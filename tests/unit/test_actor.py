"""Unit tests for Actor Agent."""

from __future__ import annotations

import struct

import numpy as np

from src.agents.actor import _encode_wav
from src.agents.schemas import DirectorOutput, EmotionTag, Segment
from src.orchestrator.state import GraphState


class TestEncodeWav:
    """Tests for WAV encoding utility."""

    def test_valid_wav_header(self) -> None:
        waveform = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)
        wav = _encode_wav(waveform, 22050)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "

    def test_correct_data_size(self) -> None:
        waveform = np.zeros(1000, dtype=np.float32)
        wav = _encode_wav(waveform, 22050)

        data_offset = wav.find(b"data")
        data_size = struct.unpack("<I", wav[data_offset + 4 : data_offset + 8])[0]
        assert data_size == 1000 * 2  # 16-bit = 2 bytes per sample

    def test_empty_waveform(self) -> None:
        wav = _encode_wav(np.array([], dtype=np.float32), 22050)
        assert wav[:4] == b"RIFF"

    def test_roundtrip_preserves_shape(self) -> None:
        """Encode then decode should preserve audio length."""
        from src.agents.actor import _decode_wav_to_array

        waveform = np.random.randn(5000).astype(np.float32) * 0.5
        wav_bytes = _encode_wav(waveform, 22050)
        decoded = _decode_wav_to_array(wav_bytes)

        assert len(decoded) == len(waveform)

    def test_sample_rate_in_header(self) -> None:
        wav = _encode_wav(np.zeros(100, dtype=np.float32), 44100)
        sample_rate = struct.unpack("<I", wav[24:28])[0]
        assert sample_rate == 44100


class TestActorState:
    """Tests for Actor state handling (without TTS model)."""

    def test_director_output_parsing(self) -> None:
        """Actor can parse DirectorOutput from state."""
        state = GraphState(
            ssml_markup=DirectorOutput(
                segments=[Segment(text="Test", emotion=EmotionTag.HAPPY)],
                voice_id="speaker_1",
            ).model_dump()
        )
        output = DirectorOutput.model_validate(state.ssml_markup)
        assert output.segments[0].emotion == EmotionTag.HAPPY
