"""Shared data structures for the ClinicalClaw research prototype."""

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


Route = Literal["parametric_memory", "atomic", "associative", "reasoning"]
ClinicalScenario = Literal[
    "biomedical_research",
    "symptom_consultation",
    "medication_guidance",
    "diagnostic_reasoning",
    "rehabilitation",
    "general",
]
QueryComplexity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class EnhancedQuery:
    """Pre-retrieval query enhancement record."""

    original_query: str
    rewritten_query: str
    expansion_terms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryPlan:
    """A clinical query plan produced before retrieval."""

    original_question: str
    subqueries: list[str]
    route: Route = "atomic"
    scenario: ClinicalScenario = "general"


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document with score metadata."""

    document: Document
    score: float
    rank: int
    retriever: str
    components: dict[str, float | str] = field(default_factory=dict)


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


CorrectionSeverity = Literal["low", "medium", "high"]
CorrectionCategory = Literal[
    "missing_citation",
    "unsupported_claim",
    "safety_risk",
    "contradiction_risk",
]


@dataclass(frozen=True)
class CorrectionFinding:
    """A post-generation clinical output audit finding."""

    finding_id: str
    category: CorrectionCategory
    severity: CorrectionSeverity
    message: str
    claim_id: str | None = None
    evidence_doc_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CorrectionReport:
    """Structured output audit and correction report."""

    original_answer: str
    corrected_answer: str
    findings: list[CorrectionFinding] = field(default_factory=list)
    requires_review: bool = False
    notes: list[str] = field(default_factory=list)


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
    correction_report: CorrectionReport | None = None
    disclaimer: str = SAFETY_DISCLAIMER


@dataclass(frozen=True)
class EvaluationSummary:
    """Named metric outputs from a research evaluation run."""

    metrics: dict[str, float]
    notes: list[str] = field(default_factory=list)
