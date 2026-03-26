"""Tests for the pronunciation cache module."""

from __future__ import annotations

import pytest

from src.agents.pronunciation_cache import PronunciationCache


class TestPronunciationCache:
    """Tests for cross-session pronunciation cache."""

    @pytest.fixture
    def cache(self) -> PronunciationCache:
        return PronunciationCache(min_success_threshold=2)

    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache: PronunciationCache) -> None:
        result = await cache.get("Moscow", "speaker_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_single_success_below_threshold(self, cache: PronunciationCache) -> None:
        await cache.record("Moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
        # One success — below threshold of 2
        result = await cache.get("Moscow", "speaker_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_hit_after_threshold(self, cache: PronunciationCache) -> None:
        await cache.record("Moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
        await cache.record("Moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
        # Two successes — meets threshold
        result = await cache.get("Moscow", "speaker_1")
        assert result == "[ˈmɒskaʊ]"

    @pytest.mark.asyncio
    async def test_voice_isolation(self, cache: PronunciationCache) -> None:
        await cache.record("Moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
        await cache.record("Moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
        # speaker_1 has hit, speaker_2 should be miss
        assert await cache.get("Moscow", "speaker_1") == "[ˈmɒskaʊ]"
        assert await cache.get("Moscow", "speaker_2") is None

    @pytest.mark.asyncio
    async def test_case_insensitive(self, cache: PronunciationCache) -> None:
        await cache.record("Moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
        await cache.record("MOSCOW", "speaker_1", "[ˈmɒskaʊ]", success=True)
        assert await cache.get("moscow", "speaker_1") == "[ˈmɒskaʊ]"

    @pytest.mark.asyncio
    async def test_get_hints_for_text(self, cache: PronunciationCache) -> None:
        await cache.record("moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)
        await cache.record("moscow", "speaker_1", "[ˈmɒskaʊ]", success=True)

        hints = await cache.get_hints_for_text("Moscow is beautiful", "speaker_1")
        assert "moscow" in hints
        assert hints["moscow"] == "[ˈmɒskaʊ]"

    def test_size_and_clear(self) -> None:
        cache = PronunciationCache()
        assert cache.size() == 0
        cache.clear()

    @pytest.mark.asyncio
    async def test_max_entries_eviction(self) -> None:
        cache = PronunciationCache(max_entries=2)
        await cache.record("word1", "v1", "h1", success=True)
        await cache.record("word2", "v1", "h2", success=True)
        await cache.record("word3", "v1", "h3", success=True)  # Evicts word1
        assert cache.size() <= 2
