"""WhisperX Microservice — HTTP API wrapper.

Endpoints:
    POST /transcribe   — WAV audio → text + word timestamps
    GET  /health       — health check
"""

from __future__ import annotations

import io
import os
import traceback
import time

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="WhisperX Service", version="0.1.0")

# Global model references
_model = None
_align_model = None
_align_metadata = None
_model_name = os.getenv("WHISPERX_MODEL", "large-v3")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_compute_type = "float16" if _device == "cuda" else "int8"


class TranscribeResult(BaseModel):
    """Transcription result with word timestamps."""

    text: str
    language: str
    words: list[dict]
    duration_seconds: float
    processing_time_seconds: float


def _get_model():
    """Lazy-load WhisperX model."""
    global _model
    if _model is None:
        import whisperx
        print(f"Loading WhisperX model: {_model_name} on {_device}")
        _model = whisperx.load_model(
            _model_name,
            _device,
            compute_type=_compute_type,
        )
        print("WhisperX model loaded.")
    return _model


def _get_align_model(language_code: str):
    """Lazy-load alignment model for a specific language."""
    global _align_model, _align_metadata
    import whisperx
    _align_model, _align_metadata = whisperx.load_align_model(
        language_code=language_code,
        device=_device,
    )
    return _align_model, _align_metadata


@app.get("/health")
async def health():
    """Health check."""
    loaded = _model is not None
    return {
        "status": "ok",
        "model": _model_name,
        "loaded": loaded,
        "device": _device,
        "gpu_memory_mb": round(
            torch.cuda.memory_allocated() / 1024 / 1024, 1
        ) if torch.cuda.is_available() else 0,
    }


@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = None,
):
    """Transcribe audio file with word-level timestamps."""
    start_time = time.monotonic()

    try:
        import whisperx
        import tempfile
        import soundfile as sf

        model = _get_model()

        # Read uploaded audio
        audio_bytes = await audio.read()

        # whisperx.load_audio needs a file path, not BytesIO
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            try:
                audio_array = whisperx.load_audio(tmp.name)
            except Exception:
                # Fallback: load via soundfile
                buf = io.BytesIO(audio_bytes)
                data, sr = sf.read(buf, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                # Resample to 16kHz if needed
                if sr != 16000:
                    import torchaudio
                    t = torch.from_numpy(data).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    t = resampler(t)
                    data = t.squeeze().numpy()
                audio_array = data

        duration = len(audio_array) / 16000.0

        # Transcribe
        result = model.transcribe(
            audio_array,
            batch_size=16,
            language=language,
        )

        detected_lang = result.get("language", language or "unknown")

        # Align for word timestamps
        try:
            align_model, align_meta = _get_align_model(detected_lang)
            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_meta,
                audio_array,
                _device,
                return_char_alignments=False,
            )
            segments = aligned.get("segments", result["segments"])
        except Exception:
            # Alignment might fail for some languages
            segments = result["segments"]

        # Extract full text and words
        full_text = " ".join(seg.get("text", "") for seg in segments).strip()
        words = []
        for seg in segments:
            for word in seg.get("words", []):
                words.append({
                    "word": word.get("word", ""),
                    "start": round(word.get("start", 0.0), 3),
                    "end": round(word.get("end", 0.0), 3),
                    "score": round(word.get("score", 0.0), 3),
                })

        elapsed = time.monotonic() - start_time

        return TranscribeResult(
            text=full_text,
            language=detected_lang,
            words=words,
            duration_seconds=round(duration, 3),
            processing_time_seconds=round(elapsed, 3),
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {e}"
        )


@app.on_event("startup")
async def startup():
    """Pre-load model on startup if PRELOAD=1."""
    if os.getenv("PRELOAD", "0") == "1":
        print("Preloading WhisperX model...")
        _get_model()
        print("WhisperX ready!")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9881"))
    uvicorn.run(app, host="0.0.0.0", port=port)
