"""FastAPI application factory.

Creates and configures the FastAPI app with routes,
middleware, and health checks.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import AppConfig, get_config
from src.log import get_logger, setup_logging

logger = get_logger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional application config. If None, loads from env.

    Returns:
        Configured FastAPI app.
    """
    if config is None:
        config = get_config()

    setup_logging(config)

    app = FastAPI(
        title="ReflexTTS",
        description="Multi-Agent System for Self-Correcting Speech Synthesis",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "reflex-tts"}

    # Voices endpoint (stub)
    @app.get("/voices")
    async def list_voices() -> dict[str, list[str]]:
        return {"voices": config.security.whitelisted_voices}

    logger.info("app_created", environment=config.environment.value)

    return app
