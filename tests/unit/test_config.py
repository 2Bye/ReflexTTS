"""Tests for application configuration."""

from __future__ import annotations

from src.config import AppConfig, Environment, get_config


class TestAppConfig:
    """Test suite for AppConfig loading and validation."""

    def test_default_config_loads(self) -> None:
        """Config loads with defaults without errors."""
        config = AppConfig()
        assert config.environment == Environment.DEV
        assert config.debug is False

    def test_vllm_defaults(self) -> None:
        """vLLM config has sensible defaults."""
        config = AppConfig()
        assert "8000" in config.vllm.base_url
        assert config.vllm.max_tokens == 4096
        assert config.vllm.temperature == 0.1
        assert config.vllm.max_retries == 5

    def test_cosyvoice_defaults(self) -> None:
        """CosyVoice3 config defaults."""
        config = AppConfig()
        assert "CosyVoice3" in config.cosyvoice.model_dir or "Fun-CosyVoice3" in config.cosyvoice.model_dir
        assert config.cosyvoice.load_vllm is True

    def test_security_defaults(self) -> None:
        """Security config has correct guard values."""
        config = AppConfig()
        assert config.security.max_text_length == 5000
        assert config.security.max_retries == 5
        assert config.security.wer_threshold_for_human_review == 0.15
        assert len(config.security.whitelisted_voices) == 3
        assert config.security.enable_pii_masking is True

    def test_tmp_dir_created(self, tmp_path: object) -> None:
        """Temp directory is created on config load."""
        config = AppConfig()
        assert config.tmp_dir.exists()

    def test_get_config_factory(self) -> None:
        """get_config() returns a valid AppConfig."""
        config = get_config()
        assert isinstance(config, AppConfig)

    def test_redis_defaults(self) -> None:
        """Redis config defaults."""
        config = AppConfig()
        assert "6379" in config.redis.url
        assert config.redis.session_ttl_seconds == 3600

    def test_logging_defaults(self) -> None:
        """Logging config defaults."""
        config = AppConfig()
        assert config.logging.level == "INFO"
        assert config.logging.service_name == "reflex-tts"
