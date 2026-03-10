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
) -> StateGraph:  # type: ignore[type-arg]
    """Build the LangGraph state machine for the ReflexTTS pipeline.

    Graph structure:
        director → actor → critic → route
        route: approved → END
               hotfix   → director (retry with phoneme hints)
               editor   → editor → critic (re-evaluate)  [M3]
               max_retries → END (needs_human_review)

    Args:
        vllm: vLLM client (shared by Director + Critic).
        tts: CosyVoice3 TTS client.
        asr: WhisperX ASR client.
        max_retries: Maximum correction loops.

    Returns:
        Compiled LangGraph StateGraph.
    """

    # ── Node functions (closures over clients) ──

    async def director_node(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs = await run_director(gs, vllm)
        return gs.model_dump()

    async def actor_node(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs = await run_actor(gs, tts)
        return gs.model_dump()

    async def critic_node(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs = await run_critic(gs, asr, vllm)
        gs.iteration += 1
        return gs.model_dump()

    async def mark_human_review(state: dict[str, Any]) -> dict[str, Any]:
        gs = GraphState.model_validate(state)
        gs.needs_human_review = True
        gs.agent_log.append(
            {"agent": "orchestrator", "action": "escalated", "detail": f"Max retries ({max_retries}) reached"}  # type: ignore[arg-type]
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

        Returns:
            "approved" - audio is acceptable, go to END
            "hotfix" - retry with pronunciation hints
            "needs_human_review" - max retries exhausted
        """
        gs = GraphState.model_validate(state)

        if gs.is_approved:
            logger.info("route_approved", iteration=gs.iteration, wer=gs.wer)
            return "approved"

        if gs.iteration >= max_retries:
            logger.warning("route_max_retries", iteration=gs.iteration, wer=gs.wer)
            return "needs_human_review"

        # Check if any errors can be fixed via hotfix
        has_hotfixable = any(
            e.can_hotfix for e in gs.errors if hasattr(e, "can_hotfix")
        )

        if has_hotfixable:
            logger.info("route_hotfix", iteration=gs.iteration)
            return "hotfix"

        # No hotfix available — for now escalate. Editor (M3) will handle this.
        logger.info("route_no_fix_available", iteration=gs.iteration)
        return "needs_human_review"

    # ── Build graph ──

    graph = StateGraph(dict)  # type: ignore[type-var]

    # Add nodes
    graph.add_node("director", director_node)  # type: ignore[type-var]
    graph.add_node("actor", actor_node)  # type: ignore[type-var]
    graph.add_node("critic", critic_node)  # type: ignore[type-var]
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
            "hotfix": "director",  # Loop back with hints
            "needs_human_review": "mark_human_review",
        },
    )

    graph.add_edge("mark_human_review", END)

    return graph
