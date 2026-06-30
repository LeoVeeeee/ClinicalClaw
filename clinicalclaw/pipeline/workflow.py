"""Linear ClinicalClaw workflow scaffold."""

from __future__ import annotations

from clinicalclaw.generation import CitationAnswerGenerator
from clinicalclaw.models import Document, FinalAnswer
from clinicalclaw.pipeline.router import QueryPlanner, QuestionRouter
from clinicalclaw.retrieval import HybridRetriever, StubReranker
from clinicalclaw.safety import SafetyPolicy
from clinicalclaw.verification import ClaimExtractor, EvidenceVerifier


class ClinicalClawPipeline:
    """Run the minimal clinical agentic RAG workflow.

    TODO: LangGraph workflow
    """

    def __init__(
        self,
        documents: list[Document],
        top_k: int = 3,
        bm25_weight: float = 0.65,
        dense_weight: float = 0.35,
    ) -> None:
        self.documents = documents
        self.top_k = top_k
        self.router = QuestionRouter()
        self.planner = QueryPlanner()
        self.retriever = HybridRetriever(
            documents,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        self.reranker = StubReranker()
        self.answer_generator = CitationAnswerGenerator()
        self.claim_extractor = ClaimExtractor()
        self.verifier = EvidenceVerifier()
        self.safety_policy = SafetyPolicy()

    def run(self, question: str) -> FinalAnswer:
        """Run a single question through the scaffold pipeline."""

        route = self.router.route(question)
        plan = self.planner.plan(question, route)
        retrieval_results = self.retriever.search(plan.subqueries[0], top_k=self.top_k)
        reranked_results = self.reranker.rerank(question, retrieval_results)
        generated = self.answer_generator.generate(question, reranked_results)
        claims = self.claim_extractor.extract(generated)
        verifications = self.verifier.verify(claims, generated.evidence)
        safety = self.safety_policy.evaluate(question, generated.answer, verifications)
        answer = generated.answer
        if safety.action == "refuse":
            answer = (
                "This looks like an emergency or crisis question. "
                "Please contact local emergency services or a qualified clinician immediately."
            )
        return FinalAnswer(
            question=question,
            answer=answer,
            citations=generated.citations,
            claims=claims,
            verifications=verifications,
            safety=safety,
        )
