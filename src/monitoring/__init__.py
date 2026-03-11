"""Prometheus metrics for ReflexTTS.

Exposes key metrics via /metrics endpoint for monitoring:
- Request counts, latency histograms
- Pipeline iteration counts, WER distribution
- GPU memory usage (when available)
- Active sessions gauge

Usage:
    from src.monitoring.metrics import METRICS, track_request
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class Histogram:
    """Simple histogram for tracking distributions."""

    name: str
    buckets: list[float] = field(
        default_factory=lambda: [
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0
        ]
    )
    _counts: dict[str, int] = field(default_factory=dict)
    _sum: float = 0.0
    _count: int = 0

    def observe(self, value: float) -> None:
        """Record a value."""
        self._sum += value
        self._count += 1
        for bucket in self.buckets:
            key = f"le_{bucket}"
            if value <= bucket:
                self._counts[key] = self._counts.get(key, 0) + 1

    def to_prometheus(self) -> list[str]:
        """Export as Prometheus text format."""
        lines: list[str] = []
        cumulative = 0
        for bucket in self.buckets:
            key = f"le_{bucket}"
            cumulative += self._counts.get(key, 0)
            lines.append(
                f'{self.name}_bucket{{le="{bucket}"}} {cumulative}'
            )
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return lines


@dataclass
class Counter:
    """Simple monotonic counter."""

    name: str
    _values: dict[str, int] = field(default_factory=dict)

    def inc(self, labels: str = "", value: int = 1) -> None:
        """Increment counter."""
        self._values[labels] = self._values.get(labels, 0) + value

    def to_prometheus(self) -> list[str]:
        """Export as Prometheus text format."""
        lines: list[str] = []
        for labels, val in self._values.items():
            if labels:
                lines.append(f"{self.name}{{{labels}}} {val}")
            else:
                lines.append(f"{self.name} {val}")
        if not self._values:
            lines.append(f"{self.name} 0")
        return lines


@dataclass
class Gauge:
    """Simple gauge (can go up and down)."""

    name: str
    _value: float = 0.0

    def set(self, value: float) -> None:
        """Set gauge value."""
        self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        self._value -= value

    def to_prometheus(self) -> list[str]:
        """Export as Prometheus text format."""
        return [f"{self.name} {self._value}"]


class MetricsRegistry:
    """Central metrics registry."""

    def __init__(self) -> None:
        # Request metrics
        self.requests_total = Counter("reflex_requests_total")
        self.request_latency = Histogram("reflex_request_latency_seconds")

        # Pipeline metrics
        self.pipeline_iterations = Histogram(
            "reflex_pipeline_iterations",
            buckets=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.pipeline_wer = Histogram(
            "reflex_pipeline_wer",
            buckets=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0],
        )
        self.pipeline_status = Counter("reflex_pipeline_status_total")

        # Active sessions
        self.active_sessions = Gauge("reflex_active_sessions")

        # Agent latency
        self.agent_latency = Histogram("reflex_agent_latency_seconds")

        # Error counts
        self.errors_total = Counter("reflex_errors_total")

    @contextmanager
    def track_request(
        self, voice_id: str = "unknown"
    ) -> Generator[None, None, None]:
        """Context manager to track request latency."""
        self.requests_total.inc(f'voice_id="{voice_id}"')
        self.active_sessions.inc()
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.request_latency.observe(elapsed)
            self.active_sessions.dec()

    def record_pipeline_result(
        self,
        *,
        status: str,
        wer: float | None = None,
        iterations: int = 0,
    ) -> None:
        """Record pipeline completion metrics."""
        self.pipeline_status.inc(f'status="{status}"')
        self.pipeline_iterations.observe(float(iterations))
        if wer is not None:
            self.pipeline_wer.observe(wer)

    def record_agent_step(
        self, agent: str, latency_seconds: float
    ) -> None:
        """Record individual agent step latency."""
        self.agent_latency.observe(latency_seconds)
        logger.debug(
            "agent_metric", agent=agent, latency=latency_seconds
        )

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.errors_total.inc(f'type="{error_type}"')

    def export(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines: list[str] = []
        all_metrics: list[Histogram | Counter | Gauge] = [
            self.requests_total,
            self.request_latency,
            self.pipeline_iterations,
            self.pipeline_wer,
            self.pipeline_status,
            self.active_sessions,
            self.agent_latency,
            self.errors_total,
        ]
        for metric in all_metrics:
            lines.extend(metric.to_prometheus())
            lines.append("")
        return "\n".join(lines)


# Global metrics instance
METRICS = MetricsRegistry()
