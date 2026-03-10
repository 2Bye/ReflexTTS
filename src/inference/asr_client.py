"""WhisperX ASR client wrapper.

Provides transcription with word-level timestamps (forced alignment)
for the Critic Agent. Uses WhisperX which combines Whisper-large-v3
with Wav2Vec2-based phoneme alignment.

Usage:
    client = ASRClient(config.whisperx)
    result = await client.transcribe(audio_array, sample_rate=22050)
    for word in result.word_timestamps:
        print(f"{word.word}: {word.start_ms} - {word.end_ms}")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import WhisperXConfig
from src.log import get_logger

logger = get_logger(__name__)


class ASRError(Exception):
    """Base exception for ASR client errors."""


class ASRModelNotLoadedError(ASRError):
    """Model has not been loaded yet."""


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


class ASRClient:
    """WhisperX ASR client with forced alignment.

    Provides transcription + word-level timestamps essential
    for the Critic/Editor pipeline (locating exact error positions).

    Attributes:
        config: WhisperX configuration.
    """

    def __init__(self, config: WhisperXConfig) -> None:
        self.config = config
        self._model: Any = None
        self._align_model: Any = None
        self._align_metadata: Any = None
        self._loaded = False

    def load_model(self) -> None:
        """Load WhisperX and alignment models into GPU.

        This is a blocking operation — call during startup.

        Raises:
            ASRError: If model loading fails.
        """
        try:
            import whisperx

            self._model = whisperx.load_model(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )

            # Pre-load alignment model for the default language
            if self.config.language:
                self._align_model, self._align_metadata = whisperx.load_align_model(
                    language_code=self.config.language,
                    device=self.config.device,
                )

            self._loaded = True
            logger.info(
                "asr_model_loaded",
                model=self.config.model_name,
                device=self.config.device,
            )
        except Exception as e:
            logger.error("asr_model_load_failed", error=str(e))
            raise ASRError(f"Failed to load WhisperX: {e}") from e

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio with word-level forced alignment.

        Args:
            audio: Audio waveform (float32 numpy array, mono).
            sample_rate: Sample rate of the input audio.
            language: Language code (auto-detect if None).

        Returns:
            TranscriptionResult with text and word timestamps.

        Raises:
            ASRModelNotLoadedError: Model not loaded.
            ASRTranscriptionError: Transcription failed.
        """
        self._ensure_loaded()

        logger.info(
            "asr_transcribe_start",
            audio_duration_s=f"{len(audio) / sample_rate:.2f}",
        )

        try:
            result = await asyncio.to_thread(
                self._transcribe_sync,
                audio=audio,
                sample_rate=sample_rate,
                language=language,
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
            raise ASRTranscriptionError(f"Transcription failed: {e}") from e

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        """Synchronous transcription + alignment. Runs in thread pool."""
        import whisperx

        assert self._model is not None

        # Resample to 16kHz if needed (WhisperX expects 16kHz)
        if sample_rate != 16000:
            import torch
            import torchaudio

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze().numpy()

        # Step 1: Transcribe
        transcript = self._model.transcribe(
            audio,
            batch_size=self.config.batch_size,
            language=language or self.config.language,
        )

        detected_language = transcript.get("language", language or "")

        # Step 2: Forced alignment (word-level timestamps)
        align_model = self._align_model
        align_metadata = self._align_metadata

        if align_model is None or (
            self.config.language and detected_language != self.config.language
        ):
            # Load alignment model for detected language
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.config.device,
            )

        aligned = whisperx.align(
            transcript["segments"],
            align_model,
            align_metadata,
            audio,
            device=self.config.device,
            return_char_alignments=False,
        )

        # Step 3: Extract word timestamps
        word_timestamps: list[WordTimestamp] = []
        for segment in aligned.get("segments", []):
            for word_info in segment.get("words", []):
                word_timestamps.append(
                    WordTimestamp(
                        word=word_info.get("word", ""),
                        start_ms=word_info.get("start", 0.0) * 1000,
                        end_ms=word_info.get("end", 0.0) * 1000,
                        score=word_info.get("score", 0.0),
                    )
                )

        # Build full text
        full_text = " ".join(w.word for w in word_timestamps)

        return TranscriptionResult(
            text=full_text,
            word_timestamps=word_timestamps,
            language=detected_language,
        )

    def _ensure_loaded(self) -> None:
        """Check that the model is loaded."""
        if not self._loaded or self._model is None:
            raise ASRModelNotLoadedError(
                "WhisperX model not loaded. Call load_model() first."
            )

    async def health_check(self) -> bool:
        """Check if ASR model is loaded."""
        return self._loaded and self._model is not None

    async def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._loaded = False
        logger.info("asr_model_unloaded")
