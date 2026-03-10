"""Model registry: health checks, warm-up, and lifecycle management.

Provides a single entry point to initialize, warm up, and check
the health of all inference backends (vLLM, CosyVoice3, WhisperX).

Usage:
    registry = ModelRegistry(config)
    await registry.initialize()      # Load all models
    status = await registry.health() # Check all models
    await registry.shutdown()        # Release resources
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import AppConfig
from src.inference.asr_client import ASRClient
from src.inference.tts_client import TTSClient
from src.inference.vllm_client import VLLMClient
from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status of all inference backends.

    Attributes:
        vllm: Whether vLLM server is reachable.
        tts: Whether CosyVoice3 model is loaded.
        asr: Whether WhisperX model is loaded.
        all_healthy: True only if all backends are healthy.
    """

    vllm: bool = False
    tts: bool = False
    asr: bool = False

    @property
    def all_healthy(self) -> bool:
        """Check if all backends are healthy."""
        return self.vllm and self.tts and self.asr

    def to_dict(self) -> dict[str, bool]:
        """Serialize to dictionary for API response."""
        return {
            "vllm": self.vllm,
            "tts": self.tts,
            "asr": self.asr,
            "all_healthy": self.all_healthy,
        }


class ModelRegistry:
    """Centralized lifecycle manager for all inference models.

    Handles initialization order, warm-up, health checks,
    and graceful shutdown.

    Attributes:
        vllm: vLLM client for LLM inference.
        tts: CosyVoice3 TTS client.
        asr: WhisperX ASR client.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.vllm = VLLMClient(config.vllm)
        self.tts = TTSClient(config.cosyvoice)
        self.asr = ASRClient(config.whisperx)

    async def initialize(self, *, skip_gpu_models: bool = False) -> HealthStatus:
        """Initialize all inference backends.

        Loads models in optimal order for GPU memory:
        1. vLLM (external server, just connect)
        2. CosyVoice3 (GPU model)
        3. WhisperX (GPU model)

        Args:
            skip_gpu_models: If True, skip loading GPU models
                (useful for testing or CPU-only environments).

        Returns:
            HealthStatus after initialization.
        """
        logger.info("registry_initializing")

        # 1. Check vLLM server connection
        vllm_ok = await self.vllm.health_check()
        if not vllm_ok:
            logger.warning("registry_vllm_not_available")

        if not skip_gpu_models:
            # 2. Load CosyVoice3 TTS
            try:
                self.tts.load_model()
            except Exception as e:
                logger.error("registry_tts_load_failed", error=str(e))

            # 3. Load WhisperX ASR
            try:
                self.asr.load_model()
            except Exception as e:
                logger.error("registry_asr_load_failed", error=str(e))

        status = await self.health()
        logger.info("registry_initialized", **status.to_dict())
        return status

    async def health(self) -> HealthStatus:
        """Check health of all backends.

        Returns:
            Current health status.
        """
        return HealthStatus(
            vllm=await self.vllm.health_check(),
            tts=await self.tts.health_check(),
            asr=await self.asr.health_check(),
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown all backends."""
        logger.info("registry_shutting_down")

        await self.vllm.close()
        await self.tts.close()
        await self.asr.close()

        logger.info("registry_shutdown_complete")
