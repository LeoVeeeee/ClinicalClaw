"""Post-generation clinical output audit baseline."""

from __future__ import annotations

from clinicalclaw.models import (
    Claim,
    CorrectionFinding,
    CorrectionReport,
    GeneratedAnswer,
    VerificationResult,
)


class ClinicalOutputAuditor:
    """Detect output issues before final safety handling.

    TODO: replace this deterministic audit with a clinical error detection,
    localization, and correction model inspired by MEDEC/MEDIQA-CORR.
    """

    _EMERGENCY_TERMS = {
        "chest pain",
        "cannot breathe",
        "can't breathe",
        "stroke",
        "overdose",
    }
    _MEDICATION_TERMS = {"dose", "dosage", "mg", "take", "prescribe", "stop taking"}
    _CONTRADICTION_TERMS = {"never", "always", "contraindicated", "not recommended"}

    def audit(
        self,
        question: str,
        generated_answer: GeneratedAnswer,
        claims: list[Claim],
        verifications: list[VerificationResult],
    ) -> CorrectionReport:
        """Return a structured clinical output audit report."""

        findings: list[CorrectionFinding] = []
        findings.extend(_missing_citation_findings(claims))
        findings.extend(_unsupported_claim_findings(verifications))
        findings.extend(self._safety_findings(question, generated_answer.answer))
        findings.extend(self._contradiction_risk_findings(generated_answer.answer))

        requires_review = any(
            finding.severity in {"medium", "high"} for finding in findings
        )
        corrected_answer = generated_answer.answer
        notes = ["No output audit findings detected."]
        if findings:
            notes = [
                "Output audit findings detected; answer retained with research safety context."
            ]
            corrected_answer = (
                f"{generated_answer.answer}\n\nResearch safety note: "
                "This output requires evidence and clinical-safety review before any real-world use."
            )

        return CorrectionReport(
            original_answer=generated_answer.answer,
            corrected_answer=corrected_answer,
            findings=findings,
            requires_review=requires_review,
            notes=notes,
        )

    def _safety_findings(self, question: str, answer: str) -> list[CorrectionFinding]:
        normalized = f"{question} {answer}".lower()
        findings: list[CorrectionFinding] = []
        if any(term in normalized for term in self._EMERGENCY_TERMS):
            findings.append(
                CorrectionFinding(
                    finding_id="safety-1",
                    category="safety_risk",
                    severity="high",
                    message="Emergency or crisis wording detected.",
                )
            )
        if any(term in normalized for term in self._MEDICATION_TERMS):
            findings.append(
                CorrectionFinding(
                    finding_id="safety-2",
                    category="safety_risk",
                    severity="medium",
                    message="Medication, dosing, or personal treatment wording detected.",
                )
            )
        return findings

    def _contradiction_risk_findings(self, answer: str) -> list[CorrectionFinding]:
        normalized = answer.lower()
        if not any(term in normalized for term in self._CONTRADICTION_TERMS):
            return []
        return [
            CorrectionFinding(
                finding_id="contradiction-1",
                category="contradiction_risk",
                severity="low",
                message="Absolute or contradiction-prone wording detected for review.",
            )
        ]


def _missing_citation_findings(claims: list[Claim]) -> list[CorrectionFinding]:
    findings: list[CorrectionFinding] = []
    for index, claim in enumerate(claims, start=1):
        if not claim.cited_doc_ids:
            findings.append(
                CorrectionFinding(
                    finding_id=f"citation-{index}",
                    category="missing_citation",
                    severity="medium",
                    message="Claim has no citation.",
                    claim_id=claim.claim_id,
                )
            )
    return findings


def _unsupported_claim_findings(
    verifications: list[VerificationResult],
) -> list[CorrectionFinding]:
    findings: list[CorrectionFinding] = []
    for index, result in enumerate(verifications, start=1):
        if result.status != "supported":
            findings.append(
                CorrectionFinding(
                    finding_id=f"support-{index}",
                    category="unsupported_claim",
                    severity="medium",
                    message=f"Claim verification status is {result.status}.",
                    claim_id=result.claim_id,
                    evidence_doc_ids=result.evidence_doc_ids,
                )
            )
    return findings
