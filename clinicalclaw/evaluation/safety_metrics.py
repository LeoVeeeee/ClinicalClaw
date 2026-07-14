"""Generation and safety metrics for ClinicalClaw experiments."""

from __future__ import annotations

from clinicalclaw.models import CorrectionReport, VerificationResult


def claim_support_rate(verifications: list[VerificationResult]) -> float:
    """Return the fraction of claims verified as supported."""

    if not verifications:
        return 0.0
    supported = sum(1 for result in verifications if result.status == "supported")
    return supported / len(verifications)


def unsafe_output_flag_rate(reports: list[CorrectionReport]) -> float:
    """Return the fraction of audited outputs requiring clinical review."""

    if not reports:
        return 0.0
    flagged = sum(1 for report in reports if report.requires_review)
    return flagged / len(reports)


def latency_ms(start_seconds: float, end_seconds: float) -> float:
    """Convert start/end timestamps into milliseconds."""

    return max(0.0, (end_seconds - start_seconds) * 1000)
