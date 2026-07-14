from __future__ import annotations

import subprocess
import sys

import pytest

from clinicalclaw.data import load_pubmedqa
from clinicalclaw.models import FinalAnswer
from clinicalclaw.retrieval import AdaptiveRetriever

pytest.importorskip("langgraph")

from clinicalclaw.agentic import ClinicalClawGraphPipeline
from clinicalclaw.agentic.workflow import ClinicalClawAgentNodes


def _documents():
    return [
        document for example in load_pubmedqa() for document in example.to_documents()
    ]


def test_graph_run_returns_final_answer() -> None:
    pipeline = ClinicalClawGraphPipeline(_documents(), top_k=2)

    final = pipeline.run("Does aspirin reduce platelet aggregation?")

    assert isinstance(final, FinalAnswer)
    assert final.citations
    assert final.safety.action in {"allow", "caution"}


def test_graph_routes_emergency_prompt_to_refusal() -> None:
    pipeline = ClinicalClawGraphPipeline(_documents(), top_k=2)

    final = pipeline.run("I have chest pain and cannot breathe. What should I do?")

    assert final.safety.action == "refuse"
    assert final.citations == []


def test_router_agent_uses_guardrail_action_for_refusal() -> None:
    nodes = ClinicalClawAgentNodes(_documents(), top_k=2)

    state = nodes.router_agent(
        {"question": "I have chest pain and cannot breathe. What should I do?"}
    )

    assert state["guardrail_action"] == "refuse"
    assert "route" not in state


def test_graph_routes_personal_advice_to_caution() -> None:
    pipeline = ClinicalClawGraphPipeline(_documents(), top_k=2)

    final = pipeline.run("What dose of aspirin should I take tonight?")

    assert final.safety.action == "caution"
    assert final.safety.risk_level == "medium"


def test_adaptive_retrieval_agent_reuses_existing_adaptive_retriever() -> None:
    nodes = ClinicalClawAgentNodes(_documents(), top_k=2)
    question = "Does aspirin reduce platelet aggregation?"
    route_state = nodes.router_agent({"question": question})
    enhanced = nodes.query_enhancement_agent({"question": question, **route_state})
    planned = nodes.planner_agent({"question": question, **route_state, **enhanced})

    retrieved = nodes.adaptive_retrieval_agent(
        {
            "question": question,
            **route_state,
            **enhanced,
            **planned,
        }
    )

    assert isinstance(nodes.retriever, AdaptiveRetriever)
    assert planned["route"] == "atomic"
    assert retrieved["retrieval_results"]


def test_agentic_demo_runs_without_real_api_keys() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/agentic_demo.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "ClinicalClaw Agentic Experiment" in completed.stdout
    assert "Research prototype only" in completed.stdout
