"""Shared state for the optional LangGraph workflow."""

from __future__ import annotations

from typing import TypedDict

from clinicalclaw.models import (
    Claim,
    CorrectionReport,
    EnhancedQuery,
    FinalAnswer,
    GeneratedAnswer,
    QueryPlan,
    RetrievalResult,
    SafetyDecision,
    VerificationResult,
)


class ClinicalClawGraphState(TypedDict, total=False):
    """State passed between ClinicalClaw graph agents.

    LangGraph nodes return partial updates to this dictionary. Keeping this
    shape explicit makes the multi-agent workflow easier to inspect.
    """

    question: str
    route: str
    guardrail_action: str
    enhanced_query: EnhancedQuery
    query_plan: QueryPlan
    retrieval_results: list[RetrievalResult]
    generated_answer: GeneratedAnswer
    claims: list[Claim]
    verifications: list[VerificationResult]
    correction_report: CorrectionReport
    safety_decision: SafetyDecision
    final_answer: FinalAnswer


def create_initial_state(question: str) -> ClinicalClawGraphState:
    """Create the minimal state required to start the graph."""

    return {"question": question}
