"""Pydantic schemas for inter-agent communication contracts.

Strict schemas enforce structured data flow between agents.
All LLM outputs are validated against these before passing downstream.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

# ── Director → Actor ──────────────────────────────────

class EmotionTag(StrEnum):
    """Available emotion/style tags for TTS instruct."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    SERIOUS = "serious"
    WHISPER = "whisper"


_VALID_EMOTIONS = {e.value for e in EmotionTag}


class Segment(BaseModel):
    """A single speech segment produced by the Director.

    Each segment represents a phrase or sentence with
    its own emotion and pause instructions.
    """

    text: str = Field(..., min_length=1, description="Text to synthesize")
    emotion: EmotionTag = Field(default=EmotionTag.NEUTRAL, description="Desired emotion/style")
    pause_before_ms: int = Field(default=0, ge=0, description="Pause before segment in ms")
    phoneme_hints: list[str] = Field(
        default_factory=list,
        description="Inline phoneme hints for tricky words, e.g. ['[j][ǐ]予']",
    )

    @model_validator(mode="before")
    @classmethod
    def _fallback_unknown_emotion(cls, data: dict) -> dict:  # type: ignore[type-arg]
        """Map unknown emotion values to 'neutral' instead of failing."""
        if isinstance(data, dict) and "emotion" in data:
            if data["emotion"] not in _VALID_EMOTIONS:
                data["emotion"] = "neutral"
        return data


class DirectorOutput(BaseModel):
    """Structured output from the Director Agent.

    Defines how the text should be spoken: which segments,
    what emotions, and any pronunciation hints.
    """

    segments: list[Segment] = Field(..., min_length=1)
    voice_id: str = Field(default="speaker_1")
    language: str = Field(default="Auto")
    notes: str = Field(default="", description="Director reasoning/notes")


# ── Critic → Orchestrator ─────────────────────────────

class ErrorSeverity(StrEnum):
    """How critical is the detected error."""

    CRITICAL = "critical"   # Wrong word, hallucination — must fix
    WARNING = "warning"     # Mispronunciation — should fix
    INFO = "info"           # Minor accent, acceptable


class CriticError(BaseModel):
    """A single error found by the Critic Agent."""

    word_expected: str = Field(..., description="What should have been said")
    word_actual: str = Field(..., description="What was actually said")
    start_ms: float = Field(..., ge=0, description="Error start time (ms)")
    end_ms: float = Field(..., ge=0, description="Error end time (ms)")
    severity: ErrorSeverity = Field(default=ErrorSeverity.WARNING)
    description: str = Field(default="")
    can_hotfix: bool = Field(
        default=False,
        description="True if fixable via pronunciation hotfix (inline pinyin/CMU)",
    )
    hotfix_hint: str = Field(
        default="",
        description="Phoneme hint for hotfix, e.g. '[sh][ip]'",
    )
    segment_index: int = Field(
        default=-1,
        description="Index of the segment this error belongs to (-1 = unknown)",
    )


class CriticOutput(BaseModel):
    """Structured output from the Critic Agent (Judge).

    Compares the target text with the ASR transcript and
    lists all differences with severity levels.
    """

    is_approved: bool = Field(..., description="True if audio is acceptable")
    errors: list[CriticError] = Field(default_factory=list)
    wer: float = Field(..., ge=0.0, le=1.0, description="Word Error Rate (0-1)")
    summary: str = Field(default="", description="Brief summary of findings")
