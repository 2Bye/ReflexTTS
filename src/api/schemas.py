"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    """Request body for POST /synthesize."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: str = Field(default="speaker_1", description="Voice ID from /voices")
    language: str = Field(default="auto", description="Language code or 'auto'")


class SynthesizeResponse(BaseModel):
    """Response for POST /synthesize."""

    session_id: str
    status: str = "processing"
    message: str = "Synthesis started"


class SessionStatus(BaseModel):
    """Response for GET /session/{id}/status."""

    session_id: str
    status: str  # queued, processing, completed, failed
    iteration: int = 0
    max_iterations: int = 3
    wer: float | None = None
    is_approved: bool = False
    needs_human_review: bool = False
    agent_log: list[dict[str, str]] = Field(default_factory=list)
    error_message: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
