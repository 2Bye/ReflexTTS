"""LangGraph state definition for the ReflexTTS orchestrator.

The GraphState is the central data structure passed between all agents
in the LangGraph state machine. It contains the full context of a
synthesis session.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorSeverity(StrEnum):
    """Severity level of a detected error."""

    CRITICAL = "critical"  # Must be fixed (wrong word, hallucination)
    WARNING = "warning"  # Should be fixed (pronunciation, intonation)
    INFO = "info"  # Minor issue (acceptable synonym, slight accent)


class DetectedError(BaseModel):
    """A single error detected by the Critic Agent."""

    word_expected: str
    word_actual: str
    start_ms: float
    end_ms: float
    severity: ErrorSeverity
    description: str = ""


class AgentLogEntry(BaseModel):
    """A single log entry from an agent's action."""

    agent: str
    action: str
    detail: str = ""
    timestamp_ms: float = 0.0


class GraphState(BaseModel):
    """Central state for the LangGraph orchestrator.

    Passed between agents, accumulating results at each step.
    """

    # ── Input ──
    text: str = ""
    voice_id: str = "speaker_1"
    trace_id: str = ""

    # ── Director output ──
    ssml_markup: dict[str, Any] = Field(default_factory=dict)
    tts_instruct: str = ""

    # ── Actor output ──
    audio_bytes: bytes = b""
    sample_rate: int = 22050

    # ── Critic output ──
    transcript: str = ""
    word_timestamps: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[DetectedError] = Field(default_factory=list)
    wer: float = 1.0

    # ── Editor output ──
    inpainted_audio_bytes: bytes = b""

    # ── Control flow ──
    iteration: int = 0
    max_retries: int = 3
    is_approved: bool = False
    needs_human_review: bool = False
    convergence_score: float = 0.0

    # ── Observability ──
    agent_log: list[AgentLogEntry] = Field(default_factory=list)
