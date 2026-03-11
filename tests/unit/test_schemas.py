"""Unit tests for agent schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.schemas import (
    CriticError,
    CriticOutput,
    DirectorOutput,
    EmotionTag,
    ErrorSeverity,
    Segment,
)


class TestSegment:
    """Tests for Segment schema."""

    def test_minimal_segment(self) -> None:
        seg = Segment(text="Hello world")
        assert seg.emotion == EmotionTag.NEUTRAL
        assert seg.pause_before_ms == 0
        assert seg.phoneme_hints == []

    def test_full_segment(self) -> None:
        seg = Segment(
            text="Привет мир",
            emotion=EmotionTag.HAPPY,
            pause_before_ms=500,
            phoneme_hints=["[p][r][i][v][e][t]"],
        )
        assert seg.emotion == EmotionTag.HAPPY
        assert seg.pause_before_ms == 500

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Segment(text="")


class TestDirectorOutput:
    """Tests for DirectorOutput schema."""

    def test_valid_output(self) -> None:
        output = DirectorOutput(
            segments=[Segment(text="Hello"), Segment(text="World")],
            voice_id="speaker_1",
        )
        assert len(output.segments) == 2
        assert output.language == "Auto"

    def test_empty_segments_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DirectorOutput(segments=[])

    def test_serialization(self) -> None:
        output = DirectorOutput(
            segments=[Segment(text="Test", emotion=EmotionTag.CALM)],
        )
        data = output.model_dump()
        assert data["segments"][0]["emotion"] == "calm"
        reconstructed = DirectorOutput.model_validate(data)
        assert reconstructed.segments[0].emotion == EmotionTag.CALM


class TestCriticOutput:
    """Tests for CriticOutput schema."""

    def test_approved(self) -> None:
        output = CriticOutput(is_approved=True, wer=0.0, summary="Perfect")
        assert output.is_approved is True
        assert output.errors == []

    def test_with_errors(self) -> None:
        output = CriticOutput(
            is_approved=False,
            wer=0.15,
            errors=[
                CriticError(
                    word_expected="king",
                    word_actual="thing",
                    start_ms=2450.0,
                    end_ms=3100.0,
                    severity=ErrorSeverity.CRITICAL,
                    can_hotfix=False,
                ),
            ],
        )
        assert len(output.errors) == 1
        assert output.errors[0].severity == ErrorSeverity.CRITICAL

    def test_hotfix_error(self) -> None:
        error = CriticError(
            word_expected="予",
            word_actual="与",
            start_ms=1000.0,
            end_ms=1500.0,
            severity=ErrorSeverity.WARNING,
            can_hotfix=True,
            hotfix_hint="[j][ǐ]",
        )
        assert error.can_hotfix is True
        assert error.hotfix_hint == "[j][ǐ]"

    def test_wer_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CriticOutput(is_approved=True, wer=1.5)
        with pytest.raises(ValidationError):
            CriticOutput(is_approved=True, wer=-0.1)

    def test_segment_index_default(self) -> None:
        """CriticError defaults segment_index to -1."""
        error = CriticError(
            word_expected="hello",
            word_actual="hallo",
            start_ms=100.0,
            end_ms=200.0,
        )
        assert error.segment_index == -1

    def test_segment_index_set(self) -> None:
        """CriticError can have explicit segment_index."""
        error = CriticError(
            word_expected="king",
            word_actual="thing",
            start_ms=2450.0,
            end_ms=3100.0,
            severity=ErrorSeverity.CRITICAL,
            segment_index=2,
        )
        assert error.segment_index == 2
