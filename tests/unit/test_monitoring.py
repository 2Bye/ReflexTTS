"""Unit tests for monitoring metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import AppConfig
from src.monitoring import MetricsRegistry


class TestCounter:
    """Tests for Counter metric."""

    def test_increment(self) -> None:
        reg = MetricsRegistry()
        reg.requests_total.inc('voice_id="speaker_1"')
        reg.requests_total.inc('voice_id="speaker_1"')
        export = "\n".join(reg.requests_total.to_prometheus())
        assert "2" in export

    def test_counter_export(self) -> None:
        reg = MetricsRegistry()
        lines = reg.requests_total.to_prometheus()
        assert len(lines) >= 1


class TestGauge:
    """Tests for Gauge metric."""

    def test_set_and_export(self) -> None:
        reg = MetricsRegistry()
        reg.active_sessions.set(5.0)
        lines = reg.active_sessions.to_prometheus()
        assert "5" in lines[0]

    def test_inc_dec(self) -> None:
        reg = MetricsRegistry()
        reg.active_sessions.inc()
        reg.active_sessions.inc()
        reg.active_sessions.dec()
        lines = reg.active_sessions.to_prometheus()
        assert "1" in lines[0]


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self) -> None:
        reg = MetricsRegistry()
        reg.request_latency.observe(0.5)
        reg.request_latency.observe(1.5)
        lines = reg.request_latency.to_prometheus()
        assert any("_count 2" in line for line in lines)
        assert any("_sum 2.0" in line for line in lines)


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_track_request(self) -> None:
        reg = MetricsRegistry()
        with reg.track_request("speaker_1"):
            pass  # Simulate quick request
        export = reg.export()
        assert "reflex_requests_total" in export

    def test_record_pipeline(self) -> None:
        reg = MetricsRegistry()
        reg.record_pipeline_result(status="completed", wer=0.01, iterations=2)
        export = reg.export()
        assert "reflex_pipeline_status_total" in export

    def test_record_error(self) -> None:
        reg = MetricsRegistry()
        reg.record_error("timeout")
        export = reg.export()
        assert "reflex_errors_total" in export

    def test_full_export(self) -> None:
        reg = MetricsRegistry()
        export = reg.export()
        assert "reflex_requests_total" in export
        assert "reflex_active_sessions" in export


class TestMetricsEndpoint:
    """Tests for /metrics HTTP endpoint."""

    def test_metrics_endpoint(self) -> None:
        config = AppConfig()
        app = create_app(config)
        client = TestClient(app)
        res = client.get("/metrics")
        assert res.status_code == 200
        assert "reflex_requests_total" in res.text
        assert "text/plain" in res.headers["content-type"]
