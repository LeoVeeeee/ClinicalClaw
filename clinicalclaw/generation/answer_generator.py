"""Citation-based answer generation stub."""

from __future__ import annotations

from clinicalclaw.llm import MockLLMProvider
from clinicalclaw.models import GeneratedAnswer, RetrievalResult


class CitationAnswerGenerator:
    """Generate short answers that cite retrieved evidence."""

    def __init__(self, llm: MockLLMProvider | None = None) -> None:
        self.llm = llm or MockLLMProvider()

    def generate(self, question: str, retrieval_results: list[RetrievalResult]) -> GeneratedAnswer:
        """Create a citation-bearing answer from retrieval results.

        A real system would prompt an LLM with evidence passages. The scaffold
        keeps this deterministic and easy to test.
        """

        evidence = [result.document for result in retrieval_results]
        citations = [document.doc_id for document in evidence]
        if not evidence:
            answer = "I do not have enough retrieved evidence to answer this research question."
            return GeneratedAnswer(question=question, answer=answer, citations=[], evidence=[])

        cited_fragments = " ".join(_format_cited_sentence(document.text, document.doc_id) for document in evidence[:2])
        prompt = f"Question: {question}\nEvidence: {cited_fragments}"
        _ = self.llm.complete(prompt)
        answer = f"Based on the retrieved research evidence, {cited_fragments}"
        return GeneratedAnswer(question=question, answer=answer, citations=citations, evidence=evidence)


def _format_cited_sentence(text: str, doc_id: str) -> str:
    sentence = text.strip().rstrip(".!?")
    return f"{sentence} [{doc_id}]."
