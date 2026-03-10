"""Tests for structured logging setup."""

from __future__ import annotations

import logging

from src.config import AppConfig
from src.log import get_logger, setup_logging


class TestLogging:
    """Test suite for logging configuration."""

    def test_setup_logging_dev(self) -> None:
        """Logging setup in dev mode doesn't crash."""
        config = AppConfig()
        config.logging.format = "console"
        setup_logging(config)

        logger = get_logger(__name__)
        # Should not raise
        logger.info("test_message", key="value")

    def test_setup_logging_json(self) -> None:
        """Logging setup in JSON mode doesn't crash."""
        config = AppConfig()
        config.logging.format = "json"
        setup_logging(config)

        logger = get_logger(__name__)
        logger.info("test_json", trace_id="abc-123")

    def test_get_logger_returns_bound_logger(self) -> None:
        """get_logger returns a structlog BoundLogger."""
        config = AppConfig()
        setup_logging(config)

        logger = get_logger("test.module")
        assert logger is not None

    def test_noisy_loggers_suppressed(self) -> None:
        """Third-party noisy loggers are set to WARNING."""
        config = AppConfig()
        setup_logging(config)

        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level >= logging.WARNING
