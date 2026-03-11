"""Session management for synthesis tasks.

In-memory session store (PoC). Production would use Redis.
Tracks synthesis progress, agent logs, and audio output.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from src.log import get_logger

logger = get_logger(__name__)


class SessionState(StrEnum):
    """Possible states of a synthesis session."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Session:
    """A synthesis session tracking pipeline progress.

    Attributes:
        session_id: Unique session identifier.
        text: Original input text.
        voice_id: Selected voice.
        status: Current session state.
        iteration: Current correction iteration.
        wer: Last known WER.
        is_approved: Whether Critic approved the audio.
        needs_human_review: Whether escalation occurred.
        agent_log: List of agent actions.
        audio_bytes: Final audio output (WAV).
        error_message: Error details if failed.
    """

    session_id: str
    text: str
    voice_id: str = "speaker_1"
    status: SessionState = SessionState.QUEUED
    iteration: int = 0
    max_iterations: int = 3
    wer: float | None = None
    is_approved: bool = False
    needs_human_review: bool = False
    agent_log: list[dict[str, str]] = field(default_factory=list)
    audio_bytes: bytes = b""
    error_message: str | None = None


class SessionStore:
    """In-memory session store (PoC).

    Thread-safe for single-worker uvicorn.
    For production, replace with Redis-backed store.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, text: str, voice_id: str = "speaker_1") -> Session:
        """Create a new synthesis session.

        Args:
            text: Input text.
            voice_id: Voice ID.

        Returns:
            New Session object.
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            text=text,
            voice_id=voice_id,
        )
        self._sessions[session_id] = session
        logger.info("session_created", session_id=session_id)
        return session

    def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def update(self, session: Session) -> None:
        """Update an existing session."""
        self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        self._sessions.pop(session_id, None)

    def count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)
