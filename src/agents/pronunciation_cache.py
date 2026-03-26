"""Cross-session pronunciation cache.

Stores phoneme hints that successfully corrected pronunciation errors.
Director uses this cache to proactively inject hints for known-difficult words,
avoiding unnecessary correction iterations.

Usage:
    cache = PronunciationCache()
    hint = await cache.get("Moscow", "speaker_1")
    await cache.record("Moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class HintRecord:
    """Record of a phoneme hint's performance.

    Attributes:
        hint: Phoneme hint string (IPA, pinyin, etc.).
        success_count: Number of times the hint resolved the error.
        fail_count: Number of times the hint did not help.
    """

    hint: str
    success_count: int = 0
    fail_count: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate of this hint."""
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def total_attempts(self) -> int:
        """Total number of attempts."""
        return self.success_count + self.fail_count


class PronunciationCache:
    """In-memory pronunciation hint cache.

    Stores (word, voice_id) → HintRecord mappings.
    A hint is considered reliable when success_count >= min_success_threshold.

    Attributes:
        min_success_threshold: Minimum successes before hint is auto-applied.
        max_entries: Maximum cache entries (LRU eviction when exceeded).
    """

    def __init__(
        self,
        min_success_threshold: int = 2,
        max_entries: int = 10_000,
    ) -> None:
        self.min_success_threshold = min_success_threshold
        self.max_entries = max_entries
        self._cache: dict[tuple[str, str], HintRecord] = {}
        self._lock = threading.Lock()

    async def get(self, word: str, voice_id: str) -> str | None:
        """Look up a reliable phoneme hint for a word+voice combination.

        Args:
            word: The word to look up.
            voice_id: The voice ID (hints may differ per voice).

        Returns:
            The phoneme hint string if reliable, None otherwise.
        """
        key = (word.lower(), voice_id)
        with self._lock:
            record = self._cache.get(key)
        if record is None:
            return None
        if record.success_count >= self.min_success_threshold:
            logger.debug(
                "pronunciation_cache_hit",
                word=word,
                voice_id=voice_id,
                hint=record.hint,
                success_rate=f"{record.success_rate:.2f}",
            )
            return record.hint
        return None

    async def record(
        self,
        word: str,
        voice_id: str,
        hint: str,
        *,
        success: bool,
    ) -> None:
        """Record the result of using a phoneme hint.

        Args:
            word: The word that was corrected.
            voice_id: The voice ID used.
            hint: The phoneme hint applied.
            success: Whether the hint resolved the pronunciation error.
        """
        key = (word.lower(), voice_id)
        with self._lock:
            if key not in self._cache:
                # Evict oldest entry if at capacity
                if len(self._cache) >= self.max_entries:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[key] = HintRecord(hint=hint)

            record = self._cache[key]
            # Update with most recent hint (may have improved)
            record.hint = hint
            if success:
                record.success_count += 1
            else:
                record.fail_count += 1

        logger.info(
            "pronunciation_cache_record",
            word=word,
            voice_id=voice_id,
            hint=hint,
            success=success,
            total_success=record.success_count,
            total_fail=record.fail_count,
        )

    async def get_hints_for_text(self, text: str, voice_id: str) -> dict[str, str]:
        """Get all reliable hints for words in the given text.

        Args:
            text: Full text to scan for known-difficult words.
            voice_id: Voice ID.

        Returns:
            Dict of word → hint for words with reliable hints.
        """
        hints: dict[str, str] = {}
        words = set(text.lower().split())
        with self._lock:
            for (cached_word, cached_voice), record in self._cache.items():
                if (
                    cached_voice == voice_id
                    and cached_word in words
                    and record.success_count >= self.min_success_threshold
                ):
                    hints[cached_word] = record.hint
        return hints

    def size(self) -> int:
        """Number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all cache entries (for testing)."""
        with self._lock:
            self._cache.clear()
