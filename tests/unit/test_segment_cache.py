"""Tests for the segment audio cache module."""

from __future__ import annotations

import pytest

from src.agents.segment_cache import SegmentCache


class TestSegmentCache:
    """Tests for cross-session segment audio cache."""

    @pytest.fixture
    def cache(self) -> SegmentCache:
        return SegmentCache(max_entries=100, ttl_seconds=0)  # No TTL for tests

    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache: SegmentCache) -> None:
        result = await cache.get("Hello", "speaker_1", "happy")
        assert result is None

    @pytest.mark.asyncio
    async def test_hit_after_put(self, cache: SegmentCache) -> None:
        audio = b"RIFF_test_audio_data"
        await cache.put("Hello", "speaker_1", "happy", audio, wer=0.0)
        result = await cache.get("Hello", "speaker_1", "happy")
        assert result == audio

    @pytest.mark.asyncio
    async def test_only_caches_perfect_wer(self, cache: SegmentCache) -> None:
        """Segments with WER > 0 should NOT be cached."""
        audio = b"RIFF_imperfect_audio"
        await cache.put("Hello", "speaker_1", "happy", audio, wer=0.1)
        result = await cache.get("Hello", "speaker_1", "happy")
        assert result is None  # Not cached

    @pytest.mark.asyncio
    async def test_key_isolation_emotion(self, cache: SegmentCache) -> None:
        audio_happy = b"RIFF_happy"
        audio_sad = b"RIFF_sad"
        await cache.put("Hello", "speaker_1", "happy", audio_happy, wer=0.0)
        await cache.put("Hello", "speaker_1", "sad", audio_sad, wer=0.0)
        assert await cache.get("Hello", "speaker_1", "happy") == audio_happy
        assert await cache.get("Hello", "speaker_1", "sad") == audio_sad

    @pytest.mark.asyncio
    async def test_key_isolation_voice(self, cache: SegmentCache) -> None:
        audio = b"RIFF_test"
        await cache.put("Hello", "speaker_1", "happy", audio, wer=0.0)
        assert await cache.get("Hello", "speaker_2", "happy") is None

    @pytest.mark.asyncio
    async def test_eviction(self) -> None:
        cache = SegmentCache(max_entries=2, ttl_seconds=0)
        await cache.put("a", "v", "e", b"1", wer=0.0)
        await cache.put("b", "v", "e", b"2", wer=0.0)
        await cache.put("c", "v", "e", b"3", wer=0.0)  # Evicts "a"
        assert cache.size() <= 2

    def test_size_and_clear(self) -> None:
        cache = SegmentCache()
        assert cache.size() == 0
        cache.clear()
