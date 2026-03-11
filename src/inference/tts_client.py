"""CosyVoice3 TTS client — HTTP microservice mode.

Calls the CosyVoice3 microservice (services/cosyvoice/) via HTTP.
No local model loading — all inference runs in a separate container.

Usage:
    client = TTSClient(config.cosyvoice)
    audio = await client.synthesize(
        text="Hello world",
        voice_id="speaker_1",
        instruct="Speak with excitement",
    )
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import CosyVoiceConfig
from src.log import get_logger

logger = get_logger(__name__)


class TTSError(Exception):
    """Base exception for TTS client errors."""


class TTSModelNotLoadedError(TTSError):
    """Remote model not available."""


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


def _wav_bytes_to_array(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Parse WAV bytes to numpy array + sample rate."""
    # Read WAV header
    if wav_bytes[:4] != b"RIFF":
        msg = "Invalid WAV data"
        raise TTSGenerationError(msg)

    # Parse fmt chunk
    fmt_offset = wav_bytes.index(b"fmt ") + 4
    _fmt_size = struct.unpack_from("<I", wav_bytes, fmt_offset)[0]
    (
        _audio_format,
        channels,
        sample_rate,
        _byte_rate,
        _block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", wav_bytes, fmt_offset + 4)

    # Parse data chunk
    data_offset = wav_bytes.index(b"data") + 4
    data_size = struct.unpack_from("<I", wav_bytes, data_offset)[0]
    raw_data = wav_bytes[data_offset + 4 : data_offset + 4 + data_size]

    # Convert to float32
    if bits_per_sample == 16:
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 32:
        samples = np.frombuffer(raw_data, dtype=np.float32)
    else:
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    # If stereo, take first channel
    if channels > 1:
        samples = samples[::channels]

    return samples, sample_rate


class TTSClient:
    """CosyVoice3 TTS client (HTTP microservice mode).

    Calls the CosyVoice3 container via HTTP instead of loading
    the model locally. This keeps the app container lightweight.

    Attributes:
        config: TTS configuration.
    """

    def __init__(self, config: CosyVoiceConfig) -> None:
        self.config = config
        self._base_url = config.base_url.rstrip("/") if config.base_url else ""
        self._loaded = False
        self._client: Any = None

    def load_model(self) -> None:
        """Check that the remote CosyVoice service is reachable.

        In microservice mode, this validates connectivity instead
        of loading a local model.
        """
        import httpx

        if not self._base_url:
            logger.warning(
                "tts_no_base_url",
                msg="COSYVOICE_BASE_URL not set, synthesize will fail",
            )
            return

        try:
            resp = httpx.get(f"{self._base_url}/health", timeout=10.0)
            data = resp.json()
            self._loaded = data.get("status") == "ok"
            self._client = httpx.AsyncClient(timeout=120.0)
            logger.info(
                "tts_service_connected",
                base_url=self._base_url,
                model=data.get("model", "unknown"),
                loaded=data.get("loaded", False),
            )
        except Exception as e:
            logger.error("tts_service_unreachable", error=str(e))
            raise TTSError(f"CosyVoice service unreachable: {e}") from e

    @property
    def sample_rate(self) -> int:
        """Get the model's output sample rate."""
        return self.config.sample_rate

    async def synthesize(
        self,
        text: str,
        voice_id: str = "speaker_1",
        *,
        instruct: str = "",
        language: str = "Auto",
    ) -> AudioResult:
        """Generate speech via CosyVoice3 microservice.

        Args:
            text: Input text to synthesize.
            voice_id: Whitelisted voice identifier.
            instruct: Optional emotion/style instruction.
            language: Target language.

        Returns:
            AudioResult with waveform and metadata.

        Raises:
            TTSModelNotLoadedError: Service not connected.
            TTSGenerationError: Generation failed.
        """
        self._ensure_loaded()
        assert self._client is not None

        voice_info = _VOICE_MAP.get(voice_id)
        if voice_info is None:
            raise TTSGenerationError(
                f"Unknown voice_id '{voice_id}'. "
                f"Available: {list(_VOICE_MAP.keys())}"
            )

        logger.info(
            "tts_synthesize_start",
            text_length=len(text),
            voice=voice_id,
            instruct=instruct[:50] if instruct else "",
        )

        try:
            resp = await self._client.post(
                f"{self._base_url}/synthesize",
                json={
                    "text": text,
                    "speaker_id": voice_info["name"],
                    "instruct": instruct,
                    "speed": 1.0,
                },
            )

            if resp.status_code != 200:
                detail = resp.text
                raise TTSGenerationError(
                    f"CosyVoice returned {resp.status_code}: {detail}"
                )

            wav_bytes = resp.content
            waveform, sr = _wav_bytes_to_array(wav_bytes)

            result = AudioResult(waveform=waveform, sample_rate=sr)
            logger.info(
                "tts_synthesize_done",
                duration_s=f"{result.duration_seconds:.2f}",
                voice=voice_id,
            )
            return result

        except TTSError:
            raise
        except Exception as e:
            logger.error("tts_synthesize_failed", error=str(e))
            raise TTSGenerationError(f"TTS generation failed: {e}") from e

    async def clone_voice(
        self,
        text: str,
        ref_audio_bytes: bytes,
        ref_text: str,
        *,
        instruct: str = "",
    ) -> AudioResult:
        """Clone voice via CosyVoice3 microservice.

        Args:
            text: Text to synthesize with the cloned voice.
            ref_audio_bytes: Reference audio WAV bytes.
            ref_text: Transcript of the reference audio.
            instruct: Optional instruction prefix.

        Returns:
            AudioResult with cloned voice waveform.
        """
        self._ensure_loaded()
        assert self._client is not None

        logger.info("tts_clone_start", text_length=len(text))

        try:
            resp = await self._client.post(
                f"{self._base_url}/clone",
                data={"text": text, "speaker_id": "cloned"},
                files={"audio": ("ref.wav", ref_audio_bytes, "audio/wav")},
            )

            if resp.status_code != 200:
                raise TTSGenerationError(
                    f"Clone returned {resp.status_code}: {resp.text}"
                )

            wav_bytes = resp.content
            waveform, sr = _wav_bytes_to_array(wav_bytes)

            result = AudioResult(waveform=waveform, sample_rate=sr)
            logger.info(
                "tts_clone_done",
                duration_s=f"{result.duration_seconds:.2f}",
            )
            return result

        except TTSError:
            raise
        except Exception as e:
            logger.error("tts_clone_failed", error=str(e))
            raise TTSGenerationError(f"Voice clone failed: {e}") from e

    def _ensure_loaded(self) -> None:
        """Check that the remote service is connected."""
        if not self._loaded or self._client is None:
            raise TTSModelNotLoadedError(
                "CosyVoice service not connected. Call load_model() first."
            )

    async def health_check(self) -> bool:
        """Check if remote CosyVoice service is healthy."""
        if not self._base_url:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return bool(resp.json().get("status") == "ok")
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._loaded = False
        logger.info("tts_client_closed")
