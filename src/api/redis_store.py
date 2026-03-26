"""Redis-backed session store.

Production session store using Redis with TTL-based expiry.
Serializes Session objects to JSON; audio_bytes stored as base64.

Usage:
    store = RedisSessionStore(config.redis)
    session = store.create("Hello", "speaker_1")
"""

from __future__ import annotations

import base64
import json
import uuid

import redis

from src.api.sessions import Session, SessionState
from src.config import RedisConfig
from src.log import get_logger

logger = get_logger(__name__)


class RedisSessionStore:
    """Redis-backed session store with TTL support.

    Attributes:
        ttl: Session time-to-live in seconds.
    """

    def __init__(self, config: RedisConfig) -> None:
        self._client = redis.Redis.from_url(
            config.url,
            decode_responses=False,
        )
        self.ttl = config.session_ttl_seconds
        logger.info(
            "redis_store_initialized",
            url=config.url,
            ttl=self.ttl,
        )

    def create(self, text: str, voice_id: str = "speaker_1") -> Session:
        """Create a new session and persist to Redis.

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
        self._save(session)
        logger.info("session_created", session_id=session_id, store="redis")
        return session

    def get(self, session_id: str) -> Session | None:
        """Get session by ID from Redis."""
        key = self._key(session_id)
        raw = self._client.get(key)
        if raw is None:
            return None
        return self._deserialize(raw)

    def update(self, session: Session) -> None:
        """Update an existing session in Redis."""
        self._save(session)

    def delete(self, session_id: str) -> None:
        """Delete a session from Redis."""
        self._client.delete(self._key(session_id))

    def count(self) -> int:
        """Count sessions (approximate — scans keys)."""
        keys = self._client.keys("reflex:session:*")
        return len(keys)

    def _key(self, session_id: str) -> str:
        """Build Redis key for a session."""
        return f"reflex:session:{session_id}"

    def _save(self, session: Session) -> None:
        """Serialize and save session to Redis with TTL."""
        data = self._serialize(session)
        self._client.setex(
            self._key(session.session_id),
            self.ttl,
            data,
        )

    @staticmethod
    def _serialize(session: Session) -> bytes:
        """Serialize Session to JSON bytes.

        audio_bytes are encoded as base64 to store in JSON.
        """
        d = {
            "session_id": session.session_id,
            "text": session.text,
            "voice_id": session.voice_id,
            "status": session.status.value,
            "iteration": session.iteration,
            "max_iterations": session.max_iterations,
            "wer": session.wer,
            "is_approved": session.is_approved,
            "needs_human_review": session.needs_human_review,
            "agent_log": session.agent_log,
            "audio_bytes": base64.b64encode(session.audio_bytes).decode("ascii"),
            "error_message": session.error_message,
        }
        return json.dumps(d).encode("utf-8")

    @staticmethod
    def _deserialize(raw: bytes) -> Session:
        """Deserialize JSON bytes to Session."""
        d = json.loads(raw)
        return Session(
            session_id=d["session_id"],
            text=d["text"],
            voice_id=d.get("voice_id", "speaker_1"),
            status=SessionState(d["status"]),
            iteration=d.get("iteration", 0),
            max_iterations=d.get("max_iterations", 3),
            wer=d.get("wer"),
            is_approved=d.get("is_approved", False),
            needs_human_review=d.get("needs_human_review", False),
            agent_log=d.get("agent_log", []),
            audio_bytes=base64.b64decode(d.get("audio_bytes", "")),
            error_message=d.get("error_message"),
        )
