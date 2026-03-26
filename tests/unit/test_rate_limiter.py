"""Tests for the rate limiter module."""

from __future__ import annotations

import time
from unittest.mock import patch

from src.api.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for sliding-window rate limiter."""

    def test_allow_within_limit(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.check("127.0.0.1") is True
        assert limiter.check("127.0.0.1") is True
        assert limiter.check("127.0.0.1") is True

    def test_block_after_limit(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check("127.0.0.1") is True
        assert limiter.check("127.0.0.1") is True
        assert limiter.check("127.0.0.1") is False  # Blocked

    def test_different_ips_independent(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("10.0.0.1") is True
        assert limiter.check("10.0.0.1") is False  # Blocked
        assert limiter.check("10.0.0.2") is True  # Different IP, still allowed

    def test_window_expiry(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.check("127.0.0.1") is True
        assert limiter.check("127.0.0.1") is False  # Blocked

        # Fast-forward past the window
        with patch("src.api.rate_limiter.time.monotonic", return_value=time.monotonic() + 2):
            assert limiter.check("127.0.0.1") is True  # Allowed again

    def test_reset(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("127.0.0.1") is True
        assert limiter.check("127.0.0.1") is False
        limiter.reset()
        assert limiter.check("127.0.0.1") is True  # Allowed after reset
