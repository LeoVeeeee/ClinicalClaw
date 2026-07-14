from __future__ import annotations

from clinicalclaw.correction import ClinicalOutputAuditor
from clinicalclaw.models import Claim, Document, GeneratedAnswer, VerificationResult


def test_output_auditor_flags_uncited_claim() -> None:
    generated = GeneratedAnswer(
        question="Does aspirin help?",
        answer="Aspirin reduces platelet aggregation.",
        citations=[],
        evidence=[],
    )
    claims = [Claim(claim_id="claim-1", text="Aspirin reduces platelet aggregation.")]
    report = ClinicalOutputAuditor().audit("Does aspirin help?", generated, claims, [])

    assert report.requires_review is True
    assert report.findings[0].category == "missing_citation"


def test_output_auditor_flags_dosage_language() -> None:
    generated = GeneratedAnswer(
        question="What dose of aspirin should I take?",
        answer="The answer requires clinical supervision.",
        citations=[],
        evidence=[],
    )

    report = ClinicalOutputAuditor().audit(generated.question, generated, [], [])

    assert any(finding.category == "safety_risk" for finding in report.findings)


def test_output_auditor_does_not_overcorrect_supported_cited_claim() -> None:
    document = Document(doc_id="doc-1", text="Aspirin reduces platelet aggregation.")
    generated = GeneratedAnswer(
        question="Does aspirin reduce platelet aggregation?",
        answer="Aspirin reduces platelet aggregation [doc-1].",
        citations=["doc-1"],
        evidence=[document],
    )
    claims = [
        Claim(
            claim_id="claim-1",
            text="Aspirin reduces platelet aggregation.",
            cited_doc_ids=["doc-1"],
        )
    ]
    verifications = [
        VerificationResult(
            claim_id="claim-1",
            status="supported",
            rationale="Supported by evidence.",
            evidence_doc_ids=["doc-1"],
            score=1.0,
        )
    ]

    report = ClinicalOutputAuditor().audit(
        generated.question, generated, claims, verifications
    )

    assert report.requires_review is False
    assert report.findings == []
    assert report.corrected_answer == generated.answer
