"""CosyVoice3 Microservice — HTTP API wrapper.

Endpoints:
    POST /synthesize     — text → WAV audio
    POST /clone          — text + ref_audio → WAV audio
    GET  /voices         — list available speaker IDs
    GET  /health         — health check

IMPORTANT: CosyVoice3 API works with FILE PATHS, not tensors/BytesIO.
All audio must be saved to temp files before passing to the model.
"""

from __future__ import annotations

import io
import os
import tempfile
import traceback

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="CosyVoice3 Service", version="0.2.0")

# Global model reference
_model = None
_model_name = os.getenv("COSYVOICE_MODEL", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
_default_prompt = os.getenv("DEFAULT_PROMPT_WAV", "/app/default_prompt.wav")


class SynthesizeRequest(BaseModel):
    """TTS synthesis request."""

    text: str
    speaker_id: str = ""
    instruct: str = ""
    speed: float = 1.0


def _get_model():
    """Lazy-load CosyVoice3 model."""
    global _model
    if _model is None:
        from cosyvoice.cli.cosyvoice import CosyVoice3
        print(f"Loading CosyVoice model: {_model_name}")
        _model = CosyVoice3(
            _model_name,
            load_trt=False,
            fp16=True,
        )
        print(f"Model loaded. sample_rate={_model.sample_rate}")
    return _model


def _collect_audio(model, output_gen):
    """Collect generator output into a single WAV Response."""
    audio_chunks = []
    for chunk in output_gen:
        audio_chunks.append(chunk["tts_speech"])

    if not audio_chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    waveform = torch.cat(audio_chunks, dim=-1)
    sample_rate = model.sample_rate

    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sample_rate, format="wav")

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Duration-Samples": str(waveform.shape[-1]),
        },
    )


@app.get("/health")
async def health():
    """Health check — returns model status."""
    loaded = _model is not None
    return {
        "status": "ok",
        "model": _model_name,
        "loaded": loaded,
        "device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
        "gpu_memory_mb": round(
            torch.cuda.memory_allocated() / 1024 / 1024, 1
        ) if torch.cuda.is_available() else 0,
    }


@app.get("/voices")
async def list_voices():
    """List available speaker IDs."""
    try:
        model = _get_model()
        voices = model.list_available_spks()
        return {"voices": voices}
    except Exception as e:
        return {"voices": [], "error": str(e)}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Synthesize text to WAV audio.

    CosyVoice3 uses instruct2 mode with a reference audio prompt.
    The default reference audio (/app/default_prompt.wav) is used
    when no specific voice cloning is needed.
    """
    try:
        model = _get_model()

        # CosyVoice3 LLM requires <|endofprompt|> marker
        # When no instruct provided, use empty instruct to avoid
        # the model speaking extra preamble text
        instruct = req.instruct.strip() if req.instruct else ""
        if "<|endofprompt|>" not in instruct:
            instruct = f"{instruct}<|endofprompt|>"

        prompt_wav = _default_prompt

        if not os.path.exists(prompt_wav):
            raise HTTPException(
                status_code=500,
                detail=f"Default prompt WAV not found: {prompt_wav}",
            )

        # CosyVoice3 API takes file path for prompt_wav
        output = model.inference_instruct2(
            req.text,
            instruct,
            prompt_wav,
            stream=False,
            speed=req.speed,
        )

        return _collect_audio(model, output)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")


@app.post("/clone")
async def clone(
    text: str = Form(...),
    speaker_id: str = Form("cloned"),
    audio: UploadFile = File(...),
):
    """Clone voice from reference audio and synthesize.

    CosyVoice3 zero_shot API expects a FILE PATH, so we save
    the uploaded audio to a temp file first.
    """
    try:
        model = _get_model()

        # Save uploaded audio to temp file (CosyVoice needs file path)
        ref_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            tmp_path = tmp.name

        try:
            # Zero-shot voice clone — pass FILE PATH
            output = model.inference_zero_shot(
                text,
                "",  # prompt text (empty = auto)
                tmp_path,
                stream=False,
            )

            return _collect_audio(model, output)
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Clone failed: {e}")


@app.on_event("startup")
async def startup():
    """Pre-load model on startup if PRELOAD=1."""
    if os.getenv("PRELOAD", "0") == "1":
        print("Preloading model...")
        _get_model()
        print("Model ready!")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9880"))
    uvicorn.run(app, host="0.0.0.0", port=port)
