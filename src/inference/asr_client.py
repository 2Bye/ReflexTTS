"""WhisperX ASR client — HTTP microservice mode.

Calls the WhisperX microservice (services/whisperx/) via HTTP.
No local model loading — inference runs in a separate container.

Usage:
    client = ASRClient(config.whisperx)
    result = await client.transcribe(audio_bytes)
    for word in result.word_timestamps:
        print(f"{word.word}: {word.start_ms} - {word.end_ms}")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import WhisperXConfig
from src.log import get_logger

logger = get_logger(__name__)


class ASRError(Exception):
    """Base exception for ASR client errors."""


class ASRModelNotLoadedError(ASRError):
    """Remote model not available."""


class ASRTranscriptionError(ASRError):
    """Transcription failed."""


@dataclass
class WordTimestamp:
    """A single word with its time boundaries.

    Attributes:
        word: The transcribed word.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        score: Alignment confidence score (0-1).
    """

    word: str
    start_ms: float
    end_ms: float
    score: float = 0.0


@dataclass
class TranscriptionResult:
    """Full transcription result with word-level timestamps.

    Attributes:
        text: Full transcribed text.
        word_timestamps: Per-word time boundaries from forced alignment.
        language: Detected or specified language.
    """

    text: str
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    language: str = ""


def _array_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy float32 array to WAV bytes."""
    # Convert to 16-bit PCM
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    data = pcm.tobytes()

    # Build WAV header
    wav = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    wav += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    wav += b"data" + struct.pack("<I", len(data)) + data
    return wav


class ASRClient:
    """WhisperX ASR client (HTTP microservice mode).

    Calls the WhisperX container via HTTP for transcription
    with word-level timestamps.

    Attributes:
        config: WhisperX configuration.
    """

    def __init__(self, config: WhisperXConfig) -> None:
        self.config = config
        self._base_url = config.base_url.rstrip("/") if config.base_url else ""
        self._loaded = False
        self._client: Any = None

    def load_model(self) -> None:
        """Check that the remote WhisperX service is reachable.

        In microservice mode, validates connectivity instead of
        loading a local model.
        """
        import httpx

        if not self._base_url:
            logger.warning(
                "asr_no_base_url",
                msg="WHISPERX_BASE_URL not set, transcribe will fail",
            )
            return

        try:
            resp = httpx.get(f"{self._base_url}/health", timeout=10.0)
            data = resp.json()
            self._loaded = data.get("status") == "ok"
            self._client = httpx.AsyncClient(timeout=120.0)
            logger.info(
                "asr_service_connected",
                base_url=self._base_url,
                model=data.get("model", "unknown"),
                loaded=data.get("loaded", False),
            )
        except Exception as e:
            logger.error("asr_service_unreachable", error=str(e))
            raise ASRError(f"WhisperX service unreachable: {e}") from e

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio via WhisperX microservice.

        Args:
            audio: Audio waveform (float32 numpy, mono).
            sample_rate: Sample rate of the input audio.
            language: Language code (auto-detect if None).

        Returns:
            TranscriptionResult with text and word timestamps.

        Raises:
            ASRModelNotLoadedError: Service not connected.
            ASRTranscriptionError: Transcription failed.
        """
        self._ensure_loaded()
        assert self._client is not None

        logger.info(
            "asr_transcribe_start",
            audio_duration_s=f"{len(audio) / sample_rate:.2f}",
        )

        try:
            # Convert numpy array to WAV bytes
            wav_bytes = _array_to_wav_bytes(audio, sample_rate)

            # Call WhisperX microservice
            files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}
            params: dict[str, str] = {}
            if language:
                params["language"] = language

            resp = await self._client.post(
                f"{self._base_url}/transcribe",
                files=files,
                params=params,
            )

            if resp.status_code != 200:
                raise ASRTranscriptionError(
                    f"WhisperX returned {resp.status_code}: {resp.text}"
                )

            data = resp.json()

            # Parse response into our types
            word_timestamps: list[WordTimestamp] = []
            for word in data.get("words", []):
                word_timestamps.append(
                    WordTimestamp(
                        word=word.get("word", ""),
                        start_ms=word.get("start", 0.0) * 1000,
                        end_ms=word.get("end", 0.0) * 1000,
                        score=word.get("score", 0.0),
                    )
                )

            result = TranscriptionResult(
                text=data.get("text", ""),
                word_timestamps=word_timestamps,
                language=data.get("language", ""),
            )

            logger.info(
                "asr_transcribe_done",
                text_length=len(result.text),
                words_count=len(result.word_timestamps),
                language=result.language,
            )
            return result

        except ASRError:
            raise
        except Exception as e:
            logger.error("asr_transcribe_failed", error=str(e))
            raise ASRTranscriptionError(
                f"Transcription failed: {e}"
            ) from e

    def _ensure_loaded(self) -> None:
        """Check that the remote service is connected."""
        if not self._loaded or self._client is None:
            raise ASRModelNotLoadedError(
                "WhisperX service not connected. Call load_model() first."
            )

    async def health_check(self) -> bool:
        """Check if remote WhisperX service is healthy."""
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
        logger.info("asr_client_closed")
