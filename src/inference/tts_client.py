"""CosyVoice3 TTS client wrapper.

Provides a clean interface over CosyVoice3 AutoModel for:
- Custom voice generation (with instruct for emotions)
- Zero-shot voice cloning (with reference audio)
- Access to internal Flow Matching module (for future inpainting)

Usage:
    client = TTSClient(config.cosyvoice)
    audio = await client.synthesize(
        text="Hello world",
        voice_id="speaker_1",
        instruct="Speak with excitement",
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import CosyVoiceConfig
from src.log import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class TTSError(Exception):
    """Base exception for TTS client errors."""


class TTSModelNotLoadedError(TTSError):
    """Model has not been loaded yet."""


class TTSGenerationError(TTSError):
    """Audio generation failed."""


@dataclass
class AudioResult:
    """Result of a TTS synthesis operation.

    Attributes:
        waveform: Audio waveform as numpy array (float32).
        sample_rate: Sample rate in Hz.
        duration_seconds: Duration of the audio in seconds.
    """

    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.waveform.size > 0:
            self.duration_seconds = float(len(self.waveform)) / self.sample_rate


# Voice ID -> CosyVoice3 speaker mapping
_VOICE_MAP: dict[str, dict[str, str]] = {
    "speaker_1": {"name": "中文女", "language": "Chinese"},
    "speaker_2": {"name": "中文男", "language": "Chinese"},
    "speaker_3": {"name": "英文女", "language": "English"},
}


class TTSClient:
    """CosyVoice3 TTS client.

    Wraps CosyVoice3 AutoModel for use by Actor and Editor agents.
    Supports custom voice, zero-shot cloning, and instruct-based control.

    Attributes:
        config: TTS configuration.
    """

    def __init__(self, config: CosyVoiceConfig) -> None:
        self.config = config
        self._model: Any = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the CosyVoice3 model into GPU memory.

        This is a blocking operation — call during startup.

        Raises:
            TTSError: If model loading fails.
        """
        try:
            # Lazy import — cosyvoice is only available in GPU environments
            import sys

            sys.path.append("third_party/Matcha-TTS")
            from cosyvoice.cli.cosyvoice import AutoModel

            self._model = AutoModel(
                model_dir=self.config.model_dir,
                load_vllm=self.config.load_vllm,
                load_trt=self.config.load_trt,
                fp16=self.config.fp16,
            )
            self._loaded = True
            logger.info(
                "tts_model_loaded",
                model_dir=self.config.model_dir,
                sample_rate=self._model.sample_rate,
            )
        except Exception as e:
            logger.error("tts_model_load_failed", error=str(e))
            raise TTSError(f"Failed to load CosyVoice3 model: {e}") from e

    @property
    def sample_rate(self) -> int:
        """Get the model's output sample rate."""
        if self._model is not None:
            return int(self._model.sample_rate)
        return self.config.sample_rate

    async def synthesize(
        self,
        text: str,
        voice_id: str = "speaker_1",
        *,
        instruct: str = "",
        language: str = "Auto",
    ) -> AudioResult:
        """Generate speech from text using a predefined voice.

        Args:
            text: Input text to synthesize.
            voice_id: Whitelisted voice identifier.
            instruct: Optional emotion/style instruction.
            language: Target language (Auto for auto-detect).

        Returns:
            AudioResult with waveform and metadata.

        Raises:
            TTSModelNotLoadedError: Model not loaded.
            TTSGenerationError: Generation failed.
        """
        self._ensure_loaded()

        voice_info = _VOICE_MAP.get(voice_id)
        if voice_info is None:
            raise TTSGenerationError(
                f"Unknown voice_id '{voice_id}'. Available: {list(_VOICE_MAP.keys())}"
            )

        lang = language if language != "Auto" else voice_info["language"]

        logger.info(
            "tts_synthesize_start",
            text_length=len(text),
            voice=voice_id,
            instruct=instruct[:50] if instruct else "",
        )

        try:
            waveform = await asyncio.to_thread(
                self._generate_sync,
                text=text,
                speaker=voice_info["name"],
                language=lang,
                instruct=instruct,
            )

            result = AudioResult(waveform=waveform, sample_rate=self.sample_rate)
            logger.info(
                "tts_synthesize_done",
                duration_s=f"{result.duration_seconds:.2f}",
                voice=voice_id,
            )
            return result

        except Exception as e:
            logger.error("tts_synthesize_failed", error=str(e))
            raise TTSGenerationError(f"TTS generation failed: {e}") from e

    async def clone_voice(
        self,
        text: str,
        ref_audio_path: str | Path,
        ref_text: str,
        *,
        instruct: str = "",
    ) -> AudioResult:
        """Generate speech using zero-shot voice cloning.

        Uses a reference audio clip to clone the speaker's voice.
        Used by Editor Agent for context-conditioned chunk repair.

        Args:
            text: Text to synthesize with the cloned voice.
            ref_audio_path: Path to reference audio file.
            ref_text: Transcript of the reference audio.
            instruct: Optional system instruction prefix.

        Returns:
            AudioResult with cloned voice waveform.

        Raises:
            TTSModelNotLoadedError: Model not loaded.
            TTSGenerationError: Generation failed.
        """
        self._ensure_loaded()

        logger.info(
            "tts_clone_start",
            text_length=len(text),
            ref_audio=str(ref_audio_path),
        )

        try:
            prompt = f"{instruct}<|endofprompt|>{ref_text}" if instruct else ref_text

            waveform = await asyncio.to_thread(
                self._clone_sync,
                text=text,
                ref_text=prompt,
                ref_audio_path=str(ref_audio_path),
            )

            result = AudioResult(waveform=waveform, sample_rate=self.sample_rate)
            logger.info(
                "tts_clone_done",
                duration_s=f"{result.duration_seconds:.2f}",
            )
            return result

        except Exception as e:
            logger.error("tts_clone_failed", error=str(e))
            raise TTSGenerationError(f"Voice clone failed: {e}") from e

    def _generate_sync(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: str,
    ) -> np.ndarray:
        """Synchronous generation using CosyVoice3 AutoModel.

        Runs in a thread pool to avoid blocking the event loop.
        """
        assert self._model is not None

        if instruct:
            # Use instruct2 mode for emotion/style control
            results = list(
                self._model.inference_instruct2(
                    text,
                    f"You are a helpful assistant. {instruct}<|endofprompt|>",
                    f"./configs/voices/{speaker}.wav",
                    stream=False,
                )
            )
        else:
            # Use custom voice mode
            results = list(
                self._model.inference_zero_shot(
                    text,
                    f"You are a helpful assistant.<|endofprompt|>{speaker}",
                    f"./configs/voices/{speaker}.wav",
                    stream=False,
                )
            )

        if not results:
            raise TTSGenerationError("CosyVoice3 returned empty results")

        # Concatenate all chunks
        import torch

        waveforms: list[torch.Tensor] = [r["tts_speech"] for r in results]
        combined = torch.cat(waveforms, dim=-1)
        return combined.squeeze().cpu().numpy().astype(np.float32)

    def _clone_sync(
        self,
        text: str,
        ref_text: str,
        ref_audio_path: str,
    ) -> np.ndarray:
        """Synchronous voice clone generation."""
        assert self._model is not None

        results = list(
            self._model.inference_zero_shot(
                text,
                ref_text,
                ref_audio_path,
                stream=False,
            )
        )

        if not results:
            raise TTSGenerationError("Voice clone returned empty results")

        import torch

        waveforms: list[torch.Tensor] = [r["tts_speech"] for r in results]
        combined = torch.cat(waveforms, dim=-1)
        return combined.squeeze().cpu().numpy().astype(np.float32)

    def _ensure_loaded(self) -> None:
        """Check that the model is loaded."""
        if not self._loaded or self._model is None:
            raise TTSModelNotLoadedError(
                "CosyVoice3 model not loaded. Call load_model() first."
            )

    async def health_check(self) -> bool:
        """Check if TTS model is loaded and functional."""
        return self._loaded and self._model is not None

    async def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._loaded = False
        logger.info("tts_model_unloaded")
