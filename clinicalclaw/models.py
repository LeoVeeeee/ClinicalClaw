"""Shared data structures for the ClinicalClaw scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SAFETY_DISCLAIMER = (
    "Research prototype only. Not medical advice, diagnosis, or treatment. "
    "Consult a qualified clinician for medical decisions."
)


@dataclass(frozen=True)
class Document:
    """A retrievable evidence unit."""

    doc_id: str
    text: str
    title: str = ""
    source: str = "unknown"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PubMedQAExample:
    """Normalized PubMedQA-style sample."""

    example_id: str
    question: str
    contexts: list[str]
    answer: str | None = None

    def to_documents(self) -> list[Document]:
        """Convert contexts into retrievable documents."""

        return [
            Document(
                doc_id=f"{self.example_id}-ctx-{index + 1}",
                title=f"{self.example_id} context {index + 1}",
                text=context,
                source="pubmedqa",
                metadata={"example_id": self.example_id},
            )
            for index, context in enumerate(self.contexts)
        ]


@dataclass(frozen=True)
class QueryPlan:
    """A small query plan produced before retrieval."""

    original_question: str
    subqueries: list[str]
    route: str = "medical_qa"


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document with score metadata."""

    document: Document
    score: float
    rank: int
    retriever: str
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAnswer:
    """Citation-bearing generated answer."""

    question: str
    answer: str
    citations: list[str]
    evidence: list[Document]


@dataclass(frozen=True)
class Claim:
    """A claim extracted from an answer."""

    claim_id: str
    text: str
    cited_doc_ids: list[str] = field(default_factory=list)


VerificationStatus = Literal["supported", "contradicted", "not_enough_evidence"]


@dataclass(frozen=True)
class VerificationResult:
    """A claim-level evidence verification decision."""

    claim_id: str
    status: VerificationStatus
    rationale: str
    evidence_doc_ids: list[str] = field(default_factory=list)
    score: float = 0.0


SafetyAction = Literal["allow", "caution", "refuse"]


@dataclass(frozen=True)
class SafetyDecision:
    """A safety guardrail decision for medical QA output."""

    action: SafetyAction
    allowed: bool
    risk_level: Literal["low", "medium", "high"]
    reasons: list[str]
    disclaimer: str = SAFETY_DISCLAIMER


@dataclass(frozen=True)
class FinalAnswer:
    """Final answer returned by the ClinicalClaw pipeline."""

    question: str
    answer: str
    citations: list[str]
    claims: list[Claim]
    verifications: list[VerificationResult]
    safety: SafetyDecision
    disclaimer: str = SAFETY_DISCLAIMER
