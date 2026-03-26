"""In-memory sliding-window rate limiter.

Limits requests per IP address using a 60-second sliding window.
Thread-safe via threading.Lock for use with multi-threaded pipeline.

Usage:
    limiter = RateLimiter(max_requests=10)
    if not limiter.check("127.0.0.1"):
        raise HTTPException(429, "Rate limit exceeded")
"""

from __future__ import annotations

import threading
import time

from src.log import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Sliding-window rate limiter per client IP.

    Attributes:
        max_requests: Maximum requests allowed per window.
        window_seconds: Size of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, client_ip: str) -> bool:
        """Check if the client IP is within rate limits.

        Args:
            client_ip: Client IP address.

        Returns:
            True if request is allowed, False if rate limit exceeded.
        """
        now = time.monotonic()
        with self._lock:
            timestamps = self._requests.get(client_ip, [])
            # Remove timestamps outside the window
            cutoff = now - self.window_seconds
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self.max_requests:
                logger.warning(
                    "rate_limit_exceeded",
                    client_ip=client_ip,
                    requests_in_window=len(timestamps),
                    limit=self.max_requests,
                )
                self._requests[client_ip] = timestamps
                return False

            timestamps.append(now)
            self._requests[client_ip] = timestamps
            return True

    def reset(self) -> None:
        """Clear all rate limit state (for testing)."""
        with self._lock:
            self._requests.clear()
