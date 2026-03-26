"""LangGraph orchestrator for the ReflexTTS pipeline.

Defines the agent graph:
  Director → Actor → Critic → [Hotfix Retry / Editor / Approve]

The graph uses GraphState as shared state between nodes.
Conditional edges route based on Critic evaluation results.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.actor import run_actor
from src.agents.critic import run_critic
from src.agents.director import run_director
from src.agents.editor import run_editor
from src.agents.pronunciation_cache import PronunciationCache
from src.agents.segment_cache import SegmentCache
from src.inference.asr_client import ASRClient
from src.inference.tts_client import TTSClient
from src.inference.vllm_client import VLLMClient
from src.log import get_logger
from src.orchestrator.state import GraphState

logger = get_logger(__name__)


def build_graph(
    vllm: VLLMClient,
    tts: TTSClient,
    asr: ASRClient,
    max_retries: int = 3,
    pronunciation_cache: PronunciationCache | None = None,
    segment_cache: SegmentCache | None = None,
) -> StateGraph:  # type: ignore[type-arg]
    """Build the LangGraph state machine for the ReflexTTS pipeline.

    Graph structure:
        director → actor → critic → route
        route: approved  → END
               hotfix    → director (retry with phoneme hints)
               editor    → editor → critic (re-evaluate)
               max_retries → END (needs_human_review)

    Args:
        vllm: vLLM client (shared by Director + Critic).
        tts: CosyVoice3 TTS client.
        asr: WhisperX ASR client.
        max_retries: Maximum correction loops.
        pronunciation_cache: Optional cross-session pronunciation cache.
        segment_cache: Optional cross-session segment audio cache.

    Returns:
        Compiled LangGraph StateGraph.
    """

    # ── Node functions (closures over clients) ──

    async def director_node(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs = await run_director(gs, vllm, pronunciation_cache=pronunciation_cache)
        return gs.model_dump()

    async def actor_node(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs = await run_actor(gs, tts, segment_cache=segment_cache)
        return gs.model_dump()

    async def critic_node(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs = await run_critic(gs, asr, vllm, pronunciation_cache=pronunciation_cache)
        gs.iteration += 1
        return gs.model_dump()

    async def editor_node(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs = await run_editor(gs, tts)
        return gs.model_dump()

    async def mark_human_review(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs.needs_human_review = True
        gs.agent_log.append(
            {  # type: ignore[arg-type]
                "agent": "orchestrator",
                "action": "escalated",
                "detail": f"Max retries ({max_retries}) reached",
            }
        )
        logger.warning(
            "pipeline_escalated",
            iteration=gs.iteration,
            wer=gs.wer,
        )
        return gs.model_dump()

    # ── Routing logic ──

    def route_after_critic(state: dict[str, Any]) -> str:
        """Decide next step based on Critic evaluation.

        Per-segment aware routing:
        - If all segments approved → END
        - If max retries → human review
        - If failed segment errors are all hotfixable → Director
        - Otherwise → Editor for targeted chunk regen

        Returns:
            "approved" - audio is acceptable, go to END
            "hotfix" - retry with pronunciation hints via Director
            "editor" - latent inpainting / chunk regen via Editor
            "needs_human_review" - max retries exhausted
        """
        gs = GraphState.model_validate(state)

        if gs.is_approved:
            logger.info("route_approved", iteration=gs.iteration, wer=gs.wer)
            return "approved"

        if gs.iteration >= max_retries:
            logger.warning("route_max_retries", iteration=gs.iteration)
            return "needs_human_review"

        # Check errors from unapproved segments only
        unapproved_errors = [
            e for e in gs.errors
            if e.segment_index < 0 or (
                e.segment_index < len(gs.segment_approved)
                and not gs.segment_approved[e.segment_index]
            )
        ]

        if not unapproved_errors:
            # No errors but not approved — safety fallback
            logger.info("route_hotfix_no_errors", iteration=gs.iteration)
            return "hotfix"

        all_hotfixable = all(
            e.can_hotfix for e in unapproved_errors
        )

        if all_hotfixable:
            logger.info(
                "route_hotfix",
                iteration=gs.iteration,
                failed_segments=[
                    i for i, a in enumerate(gs.segment_approved) if not a
                ],
            )
            return "hotfix"

        # Some errors need deeper correction → Editor
        logger.info(
            "route_editor",
            iteration=gs.iteration,
            failed_segments=[
                i for i, a in enumerate(gs.segment_approved) if not a
            ],
        )
        return "editor"

    # ── Build graph ──

    graph = StateGraph(dict)  # type: ignore[type-var]

    # Add nodes
    graph.add_node("director", director_node)  # type: ignore[type-var]
    graph.add_node("actor", actor_node)  # type: ignore[type-var]
    graph.add_node("critic", critic_node)  # type: ignore[type-var]
    graph.add_node("editor", editor_node)  # type: ignore[type-var]
    graph.add_node("mark_human_review", mark_human_review)  # type: ignore[type-var]

    # Add edges
    graph.set_entry_point("director")
    graph.add_edge("director", "actor")
    graph.add_edge("actor", "critic")

    # Conditional routing after Critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "approved": END,
            "hotfix": "director",      # Loop back with phoneme hints
            "editor": "editor",         # Latent inpainting / chunk regen
            "needs_human_review": "mark_human_review",
        },
    )

    # After Editor, re-evaluate with Critic
    graph.add_edge("editor", "critic")
    graph.add_edge("mark_human_review", END)

    return graph
