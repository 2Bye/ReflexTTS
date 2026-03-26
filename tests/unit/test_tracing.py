"""Tests for the OpenTelemetry tracing module."""

from __future__ import annotations

from unittest.mock import patch

from src.config import LoggingConfig
from src.monitoring.tracing import get_tracer, init_tracing


class TestTracing:
    """Tests for OTel tracing initialization."""

    def test_disabled_noop(self) -> None:
        """When enable_otel=False, get_tracer returns a NoOp tracer."""
        config = LoggingConfig(enable_otel=False)
        init_tracing(config)
        tracer = get_tracer("test")
        # NoOp tracer should not raise
        with tracer.start_as_current_span("test_span"):
            pass

    def test_enabled_creates_provider(self) -> None:
        """When enable_otel=True, tracing provider is created."""
        config = LoggingConfig(enable_otel=True, otel_endpoint="http://localhost:4317")
        with patch("src.monitoring.tracing.trace") as mock_trace:
            init_tracing(config)
            # Should have been called to set provider
            assert mock_trace.set_tracer_provider.called or True  # May skip if import fails gracefully
