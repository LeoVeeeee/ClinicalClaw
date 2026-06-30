"""Question routing and query planning stubs."""

from __future__ import annotations

from clinicalclaw.models import QueryPlan


class QuestionRouter:
    """Route questions into coarse workflow categories."""

    def route(self, question: str) -> str:
        lowered = question.lower()
        medical_terms = {"clinical", "patient", "disease", "treatment", "drug", "aspirin", "antibiotic"}
        if any(term in lowered for term in medical_terms):
            return "medical_qa"
        return "general_research"


class QueryPlanner:
    """Create a small query plan for retrieval."""

    def plan(self, question: str, route: str) -> QueryPlan:
        cleaned = question.strip()
        subqueries = [cleaned]
        if route == "medical_qa":
            subqueries.append(f"clinical evidence for {cleaned}")
        return QueryPlan(original_question=question, subqueries=subqueries, route=route)
