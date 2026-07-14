"""Linear ClinicalClaw baseline workflow."""

from __future__ import annotations

from clinicalclaw.correction import ClinicalOutputAuditor
from clinicalclaw.generation import CitationAnswerGenerator
from clinicalclaw.models import Document, FinalAnswer
from clinicalclaw.pipeline.router import QueryPlanner
from clinicalclaw.query_enhancement import ClinicalQueryEnhancer
from clinicalclaw.retrieval import AdaptiveRetriever, PassThroughReranker
from clinicalclaw.safety import SafetyPolicy
from clinicalclaw.verification import ClaimExtractor, EvidenceVerifier


class ClinicalClawPipeline:
    """Run the minimal clinical agentic RAG workflow."""

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

    def run(self, question: str) -> FinalAnswer:
        """Run a single question through the baseline pipeline."""

        enhanced_query = self.query_enhancer.enhance(question)
        plan = self.planner.plan(question, enhanced_query)
        retrieval_results = self.retriever.search(
            plan, enhanced_query, top_k=self.top_k
        )
        reranked_results = self.reranker.rerank(question, retrieval_results)
        generated = self.answer_generator.generate(question, reranked_results)
        claims = self.claim_extractor.extract(generated)
        verifications = self.verifier.verify(claims, generated.evidence)
        correction_report = self.output_auditor.audit(
            question, generated, claims, verifications
        )
        answer = correction_report.corrected_answer
        safety = self.safety_policy.evaluate(question, answer, verifications)
        citations = generated.citations
        if safety.action == "refuse":
            answer = (
                "This looks like an emergency or crisis question. "
                "Please contact local emergency services or a qualified clinician immediately."
            )
            citations = []
            claims = []
            verifications = []
        return FinalAnswer(
            question=question,
            answer=answer,
            citations=citations,
            claims=claims,
            verifications=verifications,
            safety=safety,
            correction_report=correction_report,
        )
