from __future__ import annotations

from clinicalclaw.safety import SafetyPolicy


def test_safety_policy_allows_research_question() -> None:
    decision = SafetyPolicy().evaluate("What evidence links aspirin and platelet aggregation?")

    assert decision.action == "allow"
    assert decision.allowed is True
    assert decision.risk_level == "low"
    assert "Not medical advice" in decision.disclaimer


def test_safety_policy_cautions_on_personal_medical_advice() -> None:
    decision = SafetyPolicy().evaluate("What dose of aspirin should I take tonight?")

    assert decision.action == "caution"
    assert decision.allowed is True
    assert decision.risk_level == "medium"


def test_safety_policy_refuses_emergency_prompt() -> None:
    decision = SafetyPolicy().evaluate("I have chest pain and cannot breathe. What should I do?")

    assert decision.action == "refuse"
    assert decision.allowed is False
    assert decision.risk_level == "high"
