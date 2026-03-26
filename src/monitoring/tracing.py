"""OpenTelemetry tracing initialization.

Configures distributed tracing when LOG_ENABLE_OTEL=true.
When disabled, all tracing operations are no-ops.

Usage:
    from src.monitoring.tracing import init_tracing, get_tracer

    init_tracing(config.logging)
    tracer = get_tracer("my_module")
    with tracer.start_as_current_span("my_operation"):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.log import get_logger

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from src.config import LoggingConfig

logger = get_logger(__name__)

_initialized = False


def init_tracing(config: LoggingConfig) -> None:
    """Initialize OpenTelemetry tracing.

    If config.enable_otel is False, this is a no-op and all subsequent
    get_tracer() calls will return a NoOp tracer.

    Args:
        config: Logging configuration with OTel settings.
    """
    global _initialized  # noqa: PLW0603

    if not config.enable_otel:
        logger.info("tracing_disabled", reason="LOG_ENABLE_OTEL=false")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": config.service_name})
        provider = TracerProvider(resource=resource)

        # Try OTLP exporter if available
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=config.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(
                "tracing_otlp_exporter",
                endpoint=config.otel_endpoint,
            )
        except ImportError:
            # OTLP exporter not installed — use console exporter for dev
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.warning(
                "tracing_console_fallback",
                message="opentelemetry-exporter-otlp not installed, using console",
            )

        trace.set_tracer_provider(provider)
        _initialized = True

        logger.info(
            "tracing_initialized",
            service_name=config.service_name,
            endpoint=config.otel_endpoint,
        )

    except Exception as e:
        logger.error(
            "tracing_init_failed",
            error=str(e),
            message="Tracing disabled due to initialization error",
        )


def get_tracer(name: str) -> Tracer:
    """Get a tracer instance.

    If tracing is not initialized, returns a NoOp tracer that
    does nothing — all span operations are safe to call.

    Args:
        name: Name of the component (e.g., module name).

    Returns:
        OpenTelemetry Tracer instance.
    """
    from opentelemetry import trace

    return trace.get_tracer(name)
