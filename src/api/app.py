"""FastAPI application factory.

Creates and configures the FastAPI app with:
- POST /synthesize — start TTS pipeline
- GET /session/{id}/status — check progress
- GET /session/{id}/audio — download audio
- GET /voices — list available voices
- GET /health — health check
- WS /ws/{session_id} — real-time agent log streaming
"""

from __future__ import annotations

import asyncio
import threading

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from src.api.schemas import ErrorResponse, SessionStatus, SynthesizeRequest, SynthesizeResponse
from src.api.sessions import SessionState, SessionStore
from src.config import AppConfig, get_config
from src.log import get_logger, setup_logging
from src.monitoring import METRICS
from src.security.input_sanitizer import sanitize_input
from src.security.pii_masker import mask_pii
from src.security.voice_whitelist import VoiceNotAllowedError, validate_voice

logger = get_logger(__name__)

# Global session store (PoC — single worker)
_store = SessionStore()

# WebSocket connections per session
_ws_connections: dict[str, list[WebSocket]] = {}

# Pipeline concurrency limiter — only 1 pipeline at a time (GPU bound)
_pipeline_semaphore = threading.Semaphore(1)


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
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "reflex-tts"}

    # ── Metrics ──
    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=METRICS.export(),
            media_type="text/plain; charset=utf-8",
        )

    # ── Voices ──
    @app.get("/voices")
    async def list_voices() -> dict[str, list[str]]:
        return {"voices": config.security.whitelisted_voices}

    # ── Synthesize ──
    @app.post("/synthesize", response_model=SynthesizeResponse, status_code=202)
    async def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
        # 1. Input sanitization
        if config.security.enable_input_sanitization:
            sanitize_result = sanitize_input(req.text, max_length=config.security.max_text_length)
            if not sanitize_result.is_safe:
                raise HTTPException(status_code=400, detail=sanitize_result.reason)
            text = sanitize_result.sanitized_text
        else:
            text = req.text

        # 2. PII masking
        if config.security.enable_pii_masking:
            pii_result = mask_pii(text)
            text = pii_result.masked_text

        # 3. Voice validation
        try:
            validate_voice(req.voice_id, config)
        except VoiceNotAllowedError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # 4. Create session
        session = _store.create(text=text, voice_id=req.voice_id)

        # 5. Launch pipeline in background thread (separate event loop)
        if not _pipeline_semaphore.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail="Pipeline busy — another synthesis is in progress. Try again later.",
            )

        t = threading.Thread(
            target=_run_pipeline_thread,
            args=(session.session_id, config),
            daemon=True,
        )
        t.start()

        return SynthesizeResponse(
            session_id=session.session_id,
            status="processing",
            message="Synthesis pipeline started",
        )

    # ── Session Status ──
    @app.get("/session/{session_id}/status", response_model=SessionStatus)
    async def session_status(session_id: str) -> SessionStatus:
        session = _store.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionStatus(
            session_id=session.session_id,
            status=session.status.value,
            iteration=session.iteration,
            max_iterations=session.max_iterations,
            wer=session.wer,
            is_approved=session.is_approved,
            needs_human_review=session.needs_human_review,
            agent_log=session.agent_log,
            error_message=session.error_message,
        )

    # ── Audio Download ──
    @app.get("/session/{session_id}/audio")
    async def session_audio(session_id: str) -> Response:
        session = _store.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.status != SessionState.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Audio not ready, status: {session.status.value}",
            )

        return Response(
            content=session.audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={session_id}.wav"},
        )

    # ── WebSocket: Agent Log Stream ──
    @app.websocket("/ws/{session_id}")
    async def websocket_log(websocket: WebSocket, session_id: str) -> None:
        session = _store.get(session_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()

        if session_id not in _ws_connections:
            _ws_connections[session_id] = []
        _ws_connections[session_id].append(websocket)

        try:
            last_log_index = 0
            while True:
                session = _store.get(session_id)
                if not session:
                    break

                # Send new log entries
                if len(session.agent_log) > last_log_index:
                    for entry in session.agent_log[last_log_index:]:
                        await websocket.send_json(entry)
                    last_log_index = len(session.agent_log)

                # Send status updates
                await websocket.send_json({
                    "type": "status",
                    "status": session.status.value,
                    "iteration": session.iteration,
                    "wer": session.wer,
                })

                if session.status in (SessionState.COMPLETED, SessionState.FAILED):
                    await websocket.send_json({
                        "type": "done",
                        "is_approved": session.is_approved,
                        "final_status": session.status.value,
                    })
                    break

                await asyncio.sleep(0.5)

        except WebSocketDisconnect:
            pass
        finally:
            if session_id in _ws_connections:
                _ws_connections[session_id] = [
                    ws for ws in _ws_connections[session_id] if ws != websocket
                ]

    # ── Web UI ──
    @app.get("/", response_class=HTMLResponse)
    async def ui() -> str:
        return _get_ui_html()

    logger.info("app_created", environment=config.environment.value)
    return app


def _run_pipeline_thread(session_id: str, config: AppConfig) -> None:
    """Thread target: runs pipeline in its own asyncio event loop.

    Completely decoupled from the main uvicorn event loop.
    Has a 300s hard timeout to prevent infinite hangs.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            asyncio.wait_for(_pipeline_impl(session_id, config), timeout=300.0)
        )
    except TimeoutError:
        logger.error("pipeline_timeout", session_id=session_id)
        session = _store.get(session_id)
        if session:
            session.status = SessionState.FAILED
            session.error_message = "Pipeline timed out (300s)"
            session.agent_log.append(
                {"agent": "orchestrator", "action": "failed", "detail": "Pipeline timed out (300s)"}
            )
            _store.update(session)
    except Exception as e:
        logger.error("pipeline_thread_error", session_id=session_id, error=str(e))
    finally:
        loop.close()
        _pipeline_semaphore.release()


async def _pipeline_impl(session_id: str, config: AppConfig) -> None:
    """Actual pipeline implementation (runs in its own event loop/thread).

    Creates inference clients, builds the LangGraph, and executes it.
    Updates session state throughout for WebSocket streaming.
    """
    session = _store.get(session_id)
    if not session:
        return

    vllm = None
    tts = None
    asr = None

    try:
        session.status = SessionState.PROCESSING
        _store.update(session)

        # ── Create inference clients ──
        from src.inference.asr_client import ASRClient
        from src.inference.tts_client import TTSClient
        from src.inference.vllm_client import VLLMClient
        from src.orchestrator.graph import build_graph
        from src.orchestrator.state import GraphState

        vllm = VLLMClient(config.vllm)
        tts = TTSClient(config.cosyvoice)
        asr = ASRClient(config.whisperx)

        # Initialize HTTP connections (sync calls are fine here — own thread)
        tts.load_model()
        asr.load_model()

        # ── Build and compile LangGraph ──
        graph = build_graph(
            vllm=vllm,
            tts=tts,
            asr=asr,
            max_retries=config.security.max_retries,
        )
        compiled = graph.compile()

        # ── Prepare initial state ──
        initial_state = GraphState(
            text=session.text,
            voice_id=session.voice_id,
            trace_id=session_id,
            max_retries=config.security.max_retries,
        ).model_dump()

        # ── Execute the graph ──
        logger.info("pipeline_start", session_id=session_id, text_length=len(session.text))

        result = await compiled.ainvoke(initial_state)

        # ── Extract results from final state ──
        final = GraphState.model_validate(result)

        session.agent_log = [
            {"agent": e.agent, "action": e.action, "detail": e.detail}
            if hasattr(e, "agent") else e
            for e in final.agent_log
        ]
        session.wer = final.wer
        session.is_approved = final.is_approved
        session.needs_human_review = final.needs_human_review
        session.iteration = final.iteration
        session.audio_bytes = final.audio_bytes
        session.status = SessionState.COMPLETED
        _store.update(session)

        logger.info(
            "pipeline_completed",
            session_id=session_id,
            wer=final.wer,
            is_approved=final.is_approved,
            iterations=final.iteration,
            audio_size_kb=len(final.audio_bytes) // 1024,
        )

    except Exception as e:
        logger.error("pipeline_failed", session_id=session_id, error=str(e))
        session.status = SessionState.FAILED
        session.error_message = str(e)
        session.agent_log.append(
            {"agent": "orchestrator", "action": "failed", "detail": str(e)}
        )
        _store.update(session)

    finally:
        # Cleanup clients
        if vllm:
            await vllm.close()
        if tts:
            await tts.close()
        if asr:
            await asr.close()


def _get_ui_html() -> str:
    """Return the embedded Web UI HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ReflexTTS</title>
<style>
:root {
  --bg: #0a0a0f;
  --surface: #12121a;
  --surface-2: #1a1a26;
  --border: #2a2a3a;
  --accent: #6c5ce7;
  --accent-glow: rgba(108,92,231,0.25);
  --green: #00d2a0;
  --red: #ff6b6b;
  --yellow: #feca57;
  --text: #e8e8f0;
  --text-dim: #8888a0;
  --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:var(--font); min-height:100vh; }
.container { max-width:860px; margin:0 auto; padding:40px 24px; }

/* Header */
.header { text-align:center; margin-bottom:48px; }
.header h1 { font-size:2.4rem; font-weight:700; background:linear-gradient(135deg,#6c5ce7,#a29bfe,#fd79a8);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.03em; }
.header p { color:var(--text-dim); margin-top:8px; font-size:0.95rem; }

/* Card */
.card { background:var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:32px; margin-bottom:24px; transition:border-color 0.3s; }
.card:hover { border-color:var(--accent); }
.card h2 { font-size:1.1rem; font-weight:600; margin-bottom:16px; display:flex; align-items:center; gap:8px; }

/* Inputs */
textarea { width:100%; min-height:120px; background:var(--surface-2); border:1px solid var(--border);
  border-radius:10px; padding:14px; color:var(--text); font-family:var(--font); font-size:0.95rem;
  resize:vertical; outline:none; transition:border 0.3s; }
textarea:focus { border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-glow); }
select { background:var(--surface-2); border:1px solid var(--border); border-radius:8px;
  padding:10px 14px; color:var(--text); font-family:var(--font); font-size:0.9rem; outline:none;
  cursor:pointer; appearance:none; min-width:160px; }
select:focus { border-color:var(--accent); }

/* Controls Row */
.controls { display:flex; align-items:center; gap:12px; margin-top:16px; flex-wrap:wrap; }

/* Buttons */
.btn { padding:12px 28px; border:none; border-radius:10px; font-family:var(--font);
  font-size:0.95rem; font-weight:600; cursor:pointer; transition:all 0.3s;
  display:inline-flex; align-items:center; gap:8px; }
.btn-primary { background:linear-gradient(135deg,#6c5ce7,#a29bfe); color:#fff;
  box-shadow:0 4px 24px var(--accent-glow); }
.btn-primary:hover { transform:translateY(-2px); box-shadow:0 8px 32px var(--accent-glow); }
.btn-primary:disabled { opacity:0.5; cursor:not-allowed; transform:none; }

/* Agent Log */
.log { background:var(--bg); border:1px solid var(--border); border-radius:10px;
  padding:16px; max-height:320px; overflow-y:auto; font-family:'JetBrains Mono',monospace;
  font-size:0.82rem; line-height:1.7; }
.log-entry { padding:4px 0; border-bottom:1px solid rgba(42,42,58,0.5); display:flex; gap:10px; }
.log-entry:last-child { border:none; }
.log-agent { color:var(--accent); font-weight:500; min-width:70px; }
.log-action { color:var(--yellow); min-width:90px; }
.log-detail { color:var(--text-dim); }

/* Status Badge */
.badge { display:inline-flex; align-items:center; gap:6px; padding:6px 14px;
  border-radius:20px; font-size:0.8rem; font-weight:500; }
.badge-processing { background:rgba(108,92,231,0.15); color:var(--accent); }
.badge-completed { background:rgba(0,210,160,0.15); color:var(--green); }
.badge-failed { background:rgba(255,107,107,0.15); color:var(--red); }
.badge-queued { background:rgba(136,136,160,0.15); color:var(--text-dim); }

/* Dot animation */
.dot { width:8px; height:8px; border-radius:50%; animation:pulse 1.5s infinite; }
.dot-processing { background:var(--accent); }
.dot-completed { background:var(--green); animation:none; }
.dot-failed { background:var(--red); animation:none; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }

/* Audio */
audio { width:100%; margin-top:16px; border-radius:8px; }

/* Progress */
.progress-bar { width:100%; height:4px; background:var(--surface-2); border-radius:2px;
  margin-top:12px; overflow:hidden; }
.progress-fill { height:100%; background:linear-gradient(90deg,#6c5ce7,#a29bfe);
  border-radius:2px; transition:width 0.5s ease; }

/* Footer */
.footer { text-align:center; color:var(--text-dim); font-size:0.8rem; margin-top:40px; }

.hidden { display:none; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>ReflexTTS</h1>
    <p>Self-Correcting Speech Synthesis with Multi-Agent Pipeline</p>
  </div>

  <div class="card">
    <h2>📝 Input</h2>
    <textarea id="textInput" placeholder="Enter text to synthesize..."
>Hello! My name is Alex.</textarea>
    <div class="controls">
      <select id="voiceSelect"></select>
      <button class="btn btn-primary" id="synthesizeBtn" onclick="synthesize()">
        ▶ Synthesize
      </button>
    </div>
  </div>

  <div class="card hidden" id="statusCard">
    <h2>
      ⚡ Pipeline
      <span class="badge badge-queued" id="statusBadge">
        <span class="dot" id="statusDot"></span>
        <span id="statusText">Queued</span>
      </span>
    </h2>
    <div class="progress-bar"><div class="progress-fill" id="progressBar" style="width:0%"></div></div>
    <div class="log" id="agentLog"></div>
  </div>

  <div class="card hidden" id="audioCard">
    <h2>🔊 Result</h2>
    <audio id="audioPlayer" controls></audio>
  </div>

  <div class="footer">
    ReflexTTS v0.1 · Director → Actor → Critic → Editor
  </div>
</div>

<script>
const API = window.location.origin;
let currentSessionId = null;
let ws = null;

// Load voices
fetch(API + '/voices').then(r => r.json()).then(data => {
  const sel = document.getElementById('voiceSelect');
  data.voices.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v; sel.appendChild(o);
  });
});

async function synthesize() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) return;

  const btn = document.getElementById('synthesizeBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Processing...';

  document.getElementById('statusCard').classList.remove('hidden');
  document.getElementById('audioCard').classList.add('hidden');
  document.getElementById('agentLog').innerHTML = '';
  setStatus('processing');

  try {
    const res = await fetch(API + '/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        voice_id: document.getElementById('voiceSelect').value,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Synthesis failed');
    }

    const data = await res.json();
    currentSessionId = data.session_id;

    // Connect WebSocket
    connectWS(currentSessionId);

  } catch (e) {
    setStatus('failed');
    addLogEntry('system', 'error', e.message);
    btn.disabled = false;
    btn.textContent = '▶ Synthesize';
  }
}

function connectWS(sessionId) {
  const wsUrl = API.replace('http', 'ws') + '/ws/' + sessionId;
  ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'status') {
      setStatus(msg.status);
      const pct = Math.min(100, (msg.iteration || 0) * 20 + (msg.wer !== null ? 60 : 30));
      document.getElementById('progressBar').style.width = pct + '%';
    } else if (msg.type === 'done') {
      setStatus(msg.final_status);
      document.getElementById('progressBar').style.width = '100%';
      if (msg.final_status === 'completed') showAudio(sessionId);
      resetButton();
    } else if (msg.agent) {
      addLogEntry(msg.agent, msg.action, msg.detail || '');
    }
  };

  ws.onerror = () => { setStatus('failed'); resetButton(); };
  ws.onclose = () => { if (currentSessionId) pollStatus(currentSessionId); };
}

async function pollStatus(sessionId) {
  const btn = document.getElementById('synthesizeBtn');
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 500));
    try {
      const res = await fetch(API + '/session/' + sessionId + '/status');
      const data = await res.json();
      setStatus(data.status);

      data.agent_log.forEach((entry, idx) => {
        if (idx >= document.getElementById('agentLog').children.length) {
          addLogEntry(entry.agent, entry.action, entry.detail || '');
        }
      });

      if (data.status === 'completed' || data.status === 'failed') {
        document.getElementById('progressBar').style.width = '100%';
        if (data.status === 'completed') showAudio(sessionId);
        resetButton();
        return;
      }
    } catch (e) { /* retry */ }
  }
  resetButton();
}

function showAudio(sessionId) {
  const card = document.getElementById('audioCard');
  card.classList.remove('hidden');
  const player = document.getElementById('audioPlayer');
  player.src = API + '/session/' + sessionId + '/audio';
}

function addLogEntry(agent, action, detail) {
  const log = document.getElementById('agentLog');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = '<span class="log-agent">' + agent
    + '</span><span class="log-action">' + action
    + '</span><span class="log-detail">' + detail + '</span>';
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function setStatus(status) {
  const badge = document.getElementById('statusBadge');
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  badge.className = 'badge badge-' + status;
  dot.className = 'dot dot-' + status;
  text.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

function resetButton() {
  const btn = document.getElementById('synthesizeBtn');
  btn.disabled = false;
  btn.textContent = '▶ Synthesize';
}
</script>
</body>
</html>"""
