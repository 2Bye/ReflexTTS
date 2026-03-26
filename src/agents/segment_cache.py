"""Cross-session segment audio cache.

Caches synthesized audio segments by (text, voice_id, emotion) to avoid
redundant GPU synthesis for repeated content.

Usage:
    cache = SegmentCache()
    cached = await cache.get("Hello world", "speaker_1", "happy")
    await cache.put("Hello world", "speaker_1", "happy", audio_bytes, wer=0.0)
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass

from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class CachedSegment:
    """A cached audio segment.

    Attributes:
        audio_bytes: WAV audio bytes.
        wer: Word Error Rate at time of caching.
        sample_rate: Audio sample rate in Hz.
        created_at: Timestamp when cached.
    """

    audio_bytes: bytes
    wer: float
    sample_rate: int
    created_at: float


class SegmentCache:
    """In-memory segment audio cache.

    Key: SHA-256( text | voice_id | emotion )
    Value: CachedSegment (audio bytes + metadata)

    Only caches segments with WER = 0.0 (perfect quality).

    Attributes:
        max_entries: Maximum number of cached segments.
        ttl_seconds: Time-to-live for cache entries (0 = no expiry).
    """

    def __init__(
        self,
        max_entries: int = 1_000,
        ttl_seconds: int = 86_400,  # 24 hours
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CachedSegment] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(text: str, voice_id: str, emotion: str) -> str:
        """Generate cache key from segment properties."""
        raw = f"{text}|{voice_id}|{emotion}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(self, text: str, voice_id: str, emotion: str) -> bytes | None:
        """Look up cached audio for a segment.

        Args:
            text: Segment text.
            voice_id: Voice ID.
            emotion: Emotion tag.

        Returns:
            Cached WAV audio bytes if found and valid, None otherwise.
        """
        key = self._make_key(text, voice_id, emotion)
        with self._lock:
            entry = self._cache.get(key)
        if entry is None:
            return None

        # Check TTL
        if self.ttl_seconds > 0:
            age = time.monotonic() - entry.created_at
            if age > self.ttl_seconds:
                with self._lock:
                    self._cache.pop(key, None)
                return None

        logger.debug(
            "segment_cache_hit",
            text_preview=text[:50],
            voice_id=voice_id,
            emotion=emotion,
            audio_size_kb=len(entry.audio_bytes) // 1024,
        )
        return entry.audio_bytes

    async def put(
        self,
        text: str,
        voice_id: str,
        emotion: str,
        audio_bytes: bytes,
        wer: float,
        sample_rate: int = 24000,
    ) -> None:
        """Cache a synthesized audio segment.

        Only segments with WER = 0.0 are cached (perfect quality).

        Args:
            text: Segment text.
            voice_id: Voice ID.
            emotion: Emotion tag.
            audio_bytes: WAV audio bytes.
            wer: Word Error Rate for this segment.
            sample_rate: Audio sample rate.
        """
        if wer > 0.0:
            return  # Only cache perfect segments

        key = self._make_key(text, voice_id, emotion)
        entry = CachedSegment(
            audio_bytes=audio_bytes,
            wer=wer,
            sample_rate=sample_rate,
            created_at=time.monotonic(),
        )

        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_entries and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = entry

        logger.info(
            "segment_cache_store",
            text_preview=text[:50],
            voice_id=voice_id,
            emotion=emotion,
            audio_size_kb=len(audio_bytes) // 1024,
        )

    def size(self) -> int:
        """Number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all cache entries (for testing)."""
        with self._lock:
            self._cache.clear()
