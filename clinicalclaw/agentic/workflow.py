"""Optional LangGraph multi-agent workflow for ClinicalClaw."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from clinicalclaw.agentic.state import ClinicalClawGraphState, create_initial_state
from clinicalclaw.correction import ClinicalOutputAuditor
from clinicalclaw.generation import CitationAnswerGenerator
from clinicalclaw.models import Document, FinalAnswer
from clinicalclaw.pipeline.router import QueryPlanner
from clinicalclaw.query_enhancement import ClinicalQueryEnhancer
from clinicalclaw.retrieval import AdaptiveRetriever, PassThroughReranker
from clinicalclaw.safety import SafetyPolicy
from clinicalclaw.verification import ClaimExtractor, EvidenceVerifier


class ClinicalClawAgentNodes:
    """Thin agent-node wrappers around the existing tested components."""

    def __init__(
        self,
        documents: list[Document],
        top_k: int = 3,
        bm25_weight: float = 0.65,
        dense_weight: float = 0.35,
    ) -> None:
        self.documents = documents
        self.top_k = top_k
        self.planner = QueryPlanner()
        self.query_enhancer = ClinicalQueryEnhancer()
        self.retriever = AdaptiveRetriever(
            documents,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        self.reranker = PassThroughReranker()
        self.answer_generator = CitationAnswerGenerator()
        self.claim_extractor = ClaimExtractor()
        self.verifier = EvidenceVerifier()
        self.output_auditor = ClinicalOutputAuditor()
        self.safety_policy = SafetyPolicy()

    def router_agent(self, state: ClinicalClawGraphState) -> ClinicalClawGraphState:
        """Classify the question before planning retrieval."""

        question = state["question"]
        safety = self.safety_policy.evaluate(question)
        if safety.action == "refuse":
            guardrail_action = "refuse"
        else:
            guardrail_action = "continue"
        return {"guardrail_action": guardrail_action, "safety_decision": safety}

    def query_enhancement_agent(
        self, state: ClinicalClawGraphState
    ) -> ClinicalClawGraphState:
        """Rewrite patient language before retrieval planning."""

        return {"enhanced_query": self.query_enhancer.enhance(state["question"])}

    def planner_agent(self, state: ClinicalClawGraphState) -> ClinicalClawGraphState:
        """Create retrieval queries for non-refused questions."""

        plan = self.planner.plan(state["question"], state.get("enhanced_query"))
        return {"query_plan": plan, "route": plan.route}

    def adaptive_retrieval_agent(
        self, state: ClinicalClawGraphState
    ) -> ClinicalClawGraphState:
        """Retrieve candidate evidence with the adaptive retriever."""

        plan = state["query_plan"]
        retrieval_results = self.retriever.search(
            plan,
            state.get("enhanced_query"),
            top_k=self.top_k,
        )
        reranked_results = self.reranker.rerank(state["question"], retrieval_results)
        return {"retrieval_results": reranked_results}

    def generator_agent(self, state: ClinicalClawGraphState) -> ClinicalClawGraphState:
        """Generate a citation-bearing answer from retrieved evidence."""

        generated = self.answer_generator.generate(
            state["question"],
            state.get("retrieval_results", []),
        )
        return {"generated_answer": generated}

    def verifier_agent(self, state: ClinicalClawGraphState) -> ClinicalClawGraphState:
        """Extract and verify answer claims against retrieved evidence."""

        generated = state["generated_answer"]
        claims = self.claim_extractor.extract(generated)
        verifications = self.verifier.verify(claims, generated.evidence)
        return {"claims": claims, "verifications": verifications}

    def correction_agent(self, state: ClinicalClawGraphState) -> ClinicalClawGraphState:
        """Audit generated output for citation, support, and safety issues."""

        report = self.output_auditor.audit(
            state["question"],
            state["generated_answer"],
            state.get("claims", []),
            state.get("verifications", []),
        )
        return {"correction_report": report}

    def safety_agent(self, state: ClinicalClawGraphState) -> ClinicalClawGraphState:
        """Apply the final safety policy and create the final answer."""

        question = state["question"]
        generated = state.get("generated_answer")
        verifications = state.get("verifications", [])
        correction_report = state.get("correction_report")
        answer_text = (
            correction_report.corrected_answer
            if correction_report is not None
            else generated.answer if generated else ""
        )
        safety = self.safety_policy.evaluate(question, answer_text, verifications)

        if safety.action == "refuse":
            final = FinalAnswer(
                question=question,
                answer=(
                    "This looks like an emergency or crisis question. "
                    "Please contact local emergency services or a qualified clinician immediately."
                ),
                citations=[],
                claims=[],
                verifications=[],
                safety=safety,
                correction_report=correction_report,
            )
            return {"safety_decision": safety, "final_answer": final}

        if generated is None:
            final = FinalAnswer(
                question=question,
                answer="I do not have enough retrieved evidence to answer this research question.",
                citations=[],
                claims=[],
                verifications=[],
                safety=safety,
                correction_report=correction_report,
            )
            return {"safety_decision": safety, "final_answer": final}

        final = FinalAnswer(
            question=question,
            answer=answer_text,
            citations=generated.citations,
            claims=state.get("claims", []),
            verifications=verifications,
            safety=safety,
            correction_report=correction_report,
        )
        return {"safety_decision": safety, "final_answer": final}


class ClinicalClawGraphPipeline:
    """Run ClinicalClaw through a LangGraph multi-agent workflow.

    TODO: connect real LangChain chat models for agent reasoning once API keys
    and model choices are configured.
    """

    def __init__(
        self,
        documents: list[Document],
        top_k: int = 3,
        bm25_weight: float = 0.65,
        dense_weight: float = 0.35,
    ) -> None:
        self.graph = build_clinicalclaw_graph(
            documents=documents,
            top_k=top_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def run(self, question: str) -> FinalAnswer:
        """Run one question through the compiled graph."""

        result = self.graph.invoke(create_initial_state(question))
        return result["final_answer"]


def build_clinicalclaw_graph(
    documents: list[Document],
    top_k: int = 3,
    bm25_weight: float = 0.65,
    dense_weight: float = 0.35,
) -> Any:
    """Build and compile the optional LangGraph workflow."""

    StateGraph, START, END = _load_langgraph()
    nodes = ClinicalClawAgentNodes(
        documents=documents,
        top_k=top_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    graph = StateGraph(ClinicalClawGraphState)
    graph.add_node("router_agent", nodes.router_agent)
    graph.add_node("query_enhancement_agent", nodes.query_enhancement_agent)
    graph.add_node("planner_agent", nodes.planner_agent)
    graph.add_node("adaptive_retrieval_agent", nodes.adaptive_retrieval_agent)
    graph.add_node("generator_agent", nodes.generator_agent)
    graph.add_node("verifier_agent", nodes.verifier_agent)
    graph.add_node("correction_agent", nodes.correction_agent)
    graph.add_node("safety_agent", nodes.safety_agent)

    graph.add_edge(START, "router_agent")
    graph.add_conditional_edges(
        "router_agent",
        _route_after_router,
        {
            "refuse": "safety_agent",
            "continue": "query_enhancement_agent",
        },
    )
    graph.add_edge("query_enhancement_agent", "planner_agent")
    graph.add_edge("planner_agent", "adaptive_retrieval_agent")
    graph.add_edge("adaptive_retrieval_agent", "generator_agent")
    graph.add_edge("generator_agent", "verifier_agent")
    graph.add_edge("verifier_agent", "correction_agent")
    graph.add_edge("correction_agent", "safety_agent")
    graph.add_edge("safety_agent", END)
    return graph.compile()


def _route_after_router(state: ClinicalClawGraphState) -> str:
    return "refuse" if state.get("guardrail_action") == "refuse" else "continue"


def _load_langgraph() -> tuple[Any, str, str]:
    _require_v1_package("langgraph")
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError(
            "ClinicalClawGraphPipeline requires optional agentic dependencies. "
            'Install them with: pip install -e ".[agentic]"'
        ) from exc
    return StateGraph, START, END


def _require_v1_package(package_name: str) -> None:
    try:
        installed_version = version(package_name)
    except PackageNotFoundError as exc:
        raise ImportError(
            "ClinicalClawGraphPipeline requires optional agentic dependencies. "
            'Install them with: pip install -e ".[agentic]"'
        ) from exc

    major = installed_version.split(".", maxsplit=1)[0]
    if major != "1":
        raise ImportError(
            f"ClinicalClawGraphPipeline requires {package_name} v1.x, "
            f'but found {installed_version}. Install with: pip install -e ".[agentic]"'
        )
