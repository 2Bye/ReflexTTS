"""Session management for synthesis tasks.

In-memory session store (PoC). Production would use Redis.
Tracks synthesis progress, agent logs, and audio output.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from src.log import get_logger

if TYPE_CHECKING:
    from src.api.redis_store import RedisSessionStore
    from src.config import AppConfig

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
        queue_position: Position in pipeline queue (None if not queued).
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
    queue_position: int | None = None


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


def create_session_store(config: AppConfig | None = None) -> SessionStore | RedisSessionStore:
    """Factory function to create the appropriate session store.

    If config.redis.use_redis is True and Redis is available, returns
    a RedisSessionStore. Otherwise falls back to in-memory SessionStore.

    Args:
        config: Application configuration.

    Returns:
        Session store instance (in-memory or Redis-backed).
    """
    if config and config.redis.use_redis:
        try:
            from src.api.redis_store import RedisSessionStore as _RedisStore

            store = _RedisStore(config.redis)
            logger.info("session_store_backend", backend="redis")
            return store
        except Exception as e:
            logger.warning(
                "redis_store_fallback",
                error=str(e),
                message="Falling back to in-memory session store",
            )

    logger.info("session_store_backend", backend="in-memory")
    return SessionStore()
