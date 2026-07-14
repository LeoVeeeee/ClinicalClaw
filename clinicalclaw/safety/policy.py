"""Safety policy baseline for research-only medical QA."""

from __future__ import annotations

from clinicalclaw.models import SAFETY_DISCLAIMER, SafetyDecision, VerificationResult


class SafetyPolicy:
    """Apply simple medical QA safety decisions."""

    _EMERGENCY_TERMS = {
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "suicide",
        "overdose",
        "stroke",
        "heart attack",
        "emergency",
    }
    _ADVICE_TERMS = {
        "should i take",
        "dose",
        "dosage",
        "prescribe",
        "treat me",
        "diagnose me",
        "stop taking",
    }

    def evaluate(
        self,
        question: str,
        answer: str = "",
        verifications: list[VerificationResult] | None = None,
    ) -> SafetyDecision:
        """Return a conservative decision for a medical QA interaction."""

        normalized = f"{question} {answer}".lower()
        reasons: list[str] = []

        if any(term in normalized for term in self._EMERGENCY_TERMS):
            reasons.append("Emergency or crisis language detected.")
            return SafetyDecision(
                action="refuse",
                allowed=False,
                risk_level="high",
                reasons=reasons,
                disclaimer=SAFETY_DISCLAIMER,
            )

        if any(term in normalized for term in self._ADVICE_TERMS):
            reasons.append("Personalized medical advice or dosing language detected.")
            return SafetyDecision(
                action="caution",
                allowed=True,
                risk_level="medium",
                reasons=reasons,
                disclaimer=SAFETY_DISCLAIMER,
            )

        if verifications and any(
            result.status != "supported" for result in verifications
        ):
            reasons.append(
                "One or more claims are not fully supported by retrieved evidence."
            )
            return SafetyDecision(
                action="caution",
                allowed=True,
                risk_level="medium",
                reasons=reasons,
                disclaimer=SAFETY_DISCLAIMER,
            )

        return SafetyDecision(
            action="allow",
            allowed=True,
            risk_level="low",
            reasons=[
                "Research-oriented question with no emergency or personalized advice trigger."
            ],
            disclaimer=SAFETY_DISCLAIMER,
        )
