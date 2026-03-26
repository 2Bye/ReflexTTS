"""Application configuration via Pydantic Settings.

All configuration is loaded from environment variables and/or .env files.
No hardcoded secrets or magic strings.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment."""

    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class VLLMConfig(BaseSettings):
    """vLLM server configuration for Qwen3-8B."""

    model_config = SettingsConfigDict(env_prefix="VLLM_")

    base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-8B-AWQ"
    api_key: str = "not-needed"  # Local vLLM doesn't require auth
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: int = 300
    max_retries: int = 5


class CosyVoiceConfig(BaseSettings):
    """CosyVoice3 TTS configuration."""

    model_config = SettingsConfigDict(env_prefix="COSYVOICE_")

    base_url: str = "http://localhost:9880"
    model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B"
    load_vllm: bool = True
    load_trt: bool = True
    fp16: bool = False
    sample_rate: int = 24000


class WhisperXConfig(BaseSettings):
    """WhisperX ASR configuration."""

    model_config = SettingsConfigDict(env_prefix="WHISPERX_")

    base_url: str = "http://localhost:9881"
    model_name: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16
    language: str | None = None  # Auto-detect if None


class SecurityConfig(BaseSettings):
    """Security and governance configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    max_text_length: int = 5000
    max_retries: int = 5
    wer_threshold_for_human_review: float = 0.15
    whitelisted_voices: list[str] = Field(
        default=["speaker_1", "speaker_2", "speaker_3"],
    )
    enable_pii_masking: bool = True
    enable_input_sanitization: bool = True


class RedisConfig(BaseSettings):
    """Redis configuration for session management."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 3600  # 1 hour
    use_redis: bool = False  # Set to True to use Redis-backed session store


class LoggingConfig(BaseSettings):
    """Logging and observability configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = "INFO"
    format: str = "json"  # "json" for production, "console" for dev
    enable_otel: bool = False
    otel_endpoint: str = "http://localhost:4317"
    service_name: str = "reflex-tts"


class APIConfig(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8080
    workers: int = 1
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    rate_limit_per_minute: int = 10


class AppConfig(BaseSettings):
    """Root application configuration.

    Aggregates all sub-configurations.
    Loads from .env file if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Environment = Environment.DEV
    debug: bool = False

    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    cosyvoice: CosyVoiceConfig = Field(default_factory=CosyVoiceConfig)
    whisperx: WhisperXConfig = Field(default_factory=WhisperXConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Paths
    tmp_dir: Path = Path("/tmp/reflex-tts")  # noqa: S108

    @field_validator("tmp_dir")
    @classmethod
    def ensure_tmp_dir_exists(cls, v: Path) -> Path:
        """Create temp directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


def get_config() -> AppConfig:
    """Factory function to create and return the application configuration.

    Returns:
        AppConfig: The fully loaded application configuration.
    """
    return AppConfig()
