"""Unit tests for LangGraph orchestrator."""

from __future__ import annotations

from src.config import AppConfig
from src.inference.asr_client import ASRClient
from src.inference.tts_client import TTSClient
from src.inference.vllm_client import VLLMClient
from src.orchestrator.graph import build_graph
from src.orchestrator.state import GraphState


class TestBuildGraph:
    """Tests for graph construction."""

    def test_graph_compiles(self) -> None:
        """Graph builds without errors."""
        config = AppConfig()
        vllm = VLLMClient(config.vllm)
        tts = TTSClient(config.cosyvoice)
        asr = ASRClient(config.whisperx)

        graph = build_graph(vllm, tts, asr, max_retries=3)
        assert graph is not None

    def test_graph_has_nodes(self) -> None:
        """Graph contains all expected nodes."""
        config = AppConfig()
        vllm = VLLMClient(config.vllm)
        tts = TTSClient(config.cosyvoice)
        asr = ASRClient(config.whisperx)

        graph = build_graph(vllm, tts, asr)
        node_names = set(graph.nodes.keys())

        assert "director" in node_names
        assert "actor" in node_names
        assert "critic" in node_names
        assert "mark_human_review" in node_names


class TestGraphState:
    """Tests for GraphState model."""

    def test_default_state(self) -> None:
        state = GraphState()
        assert state.iteration == 0
        assert state.is_approved is False
        assert state.needs_human_review is False
        assert state.max_retries == 3

    def test_serialization_roundtrip(self) -> None:
        state = GraphState(
            text="Hello", voice_id="speaker_2", iteration=1, wer=0.05
        )
        data = state.model_dump()
        restored = GraphState.model_validate(data)
        assert restored.text == "Hello"
        assert restored.voice_id == "speaker_2"
        assert restored.wer == 0.05

    def test_segment_audio_defaults(self) -> None:
        state = GraphState()
        assert state.segment_audio == []
        assert state.segment_approved == []

    def test_segment_tracking_roundtrip(self) -> None:
        state = GraphState(
            text="Hello world",
            segment_audio=[b"wav1", b"wav2"],
            segment_approved=[True, False],
        )
        data = state.model_dump()
        restored = GraphState.model_validate(data)
        assert len(restored.segment_audio) == 2
        assert restored.segment_approved == [True, False]

    def test_detected_error_segment_index(self) -> None:
        from src.orchestrator.state import DetectedError, ErrorSeverity
        err = DetectedError(
            word_expected="king",
            word_actual="thing",
            start_ms=100,
            end_ms=200,
            severity=ErrorSeverity.CRITICAL,
            segment_index=1,
        )
        assert err.segment_index == 1
