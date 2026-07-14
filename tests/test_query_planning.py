from __future__ import annotations

from collections.abc import Iterable

from clinicalclaw.pipeline.router import QueryPlanner
from clinicalclaw.query_enhancement import ClinicalQueryEnhancer
from clinicalclaw.safety import SafetyPolicy


class FakeLLMClient:
    def __init__(self, responses: Iterable[str]) -> None:
        self.responses = iter(responses)
        self.messages: list[tuple[str, str]] = []

    def complete(self, system_message: str, user_message: str) -> str:
        self.messages.append((system_message, user_message))
        return next(self.responses)


def test_simple_factual_query_routes_to_atomic_retrieval() -> None:
    question = "Does aspirin reduce platelet aggregation?"
    plan = QueryPlanner(enable_llm=False).plan(
        question, ClinicalQueryEnhancer().enhance(question)
    )

    assert plan.scenario == "biomedical_research"
    assert plan.route == "atomic"
    assert not hasattr(plan, "complexity")


def test_colloquial_query_gets_medical_terminology_enhancement() -> None:
    enhanced = ClinicalQueryEnhancer().enhance("Can antibiotics treat the flu?")

    assert enhanced.rewritten_query == "Can antibiotics treat the influenza?"
    assert "influenza" in enhanced.expansion_terms


def test_colloquial_medication_query_gets_clinical_scenario() -> None:
    question = "Can antibiotics treat the flu?"
    plan = QueryPlanner(enable_llm=False).plan(
        question, ClinicalQueryEnhancer().enhance(question)
    )

    assert plan.scenario == "medication_guidance"


def test_multi_hop_query_routes_to_associative_retrieval() -> None:
    question = "What is the relationship between aspirin and platelet aggregation?"
    plan = QueryPlanner(enable_llm=False).plan(
        question, ClinicalQueryEnhancer().enhance(question)
    )

    assert plan.route == "associative"
    assert len(plan.subqueries) >= 2


def test_reasoning_query_routes_to_reasoning_retrieval() -> None:
    question = "Why does aspirin reduce platelet aggregation?"
    plan = QueryPlanner(enable_llm=False).plan(
        question, ClinicalQueryEnhancer().enhance(question)
    )

    assert plan.route == "reasoning"


def test_general_greeting_routes_to_parametric_memory() -> None:
    question = "Hello, what can you do?"
    plan = QueryPlanner(enable_llm=False).plan(
        question, ClinicalQueryEnhancer().enhance(question)
    )

    assert plan.scenario == "general"
    assert plan.route == "parametric_memory"


def test_auto_llm_without_api_key_uses_rule_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CLINICALCLAW_DISABLE_DOTENV", "1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    plan = QueryPlanner(enable_llm="auto").plan(
        "Does aspirin reduce platelet aggregation?",
        ClinicalQueryEnhancer().enhance("Does aspirin reduce platelet aggregation?"),
    )

    assert plan.scenario == "biomedical_research"
    assert plan.route == "atomic"


def test_llm_planner_uses_valid_mock_outputs() -> None:
    fake_llm = FakeLLMClient(
        [
            '{"complexity": "high", "rationale": "asks why"}',
            '{"scenario": "diagnostic_reasoning", "rationale": "diagnosis style"}',
            (
                '{"route": "reasoning", "rationale": "needs explanation", '
                '"suggested_subqueries": ["diagnostic reasoning evidence"]}'
            ),
        ]
    )

    plan = QueryPlanner(enable_llm=True, llm_client=fake_llm).plan(
        "Why could these symptoms indicate a disease?",
        ClinicalQueryEnhancer().enhance("Why could these symptoms indicate a disease?"),
    )

    assert plan.scenario == "diagnostic_reasoning"
    assert plan.route == "reasoning"
    assert "diagnostic reasoning evidence" in plan.subqueries
    assert len(fake_llm.messages) == 3
    assert fake_llm.messages[0][0].startswith("You are a query complexity classifier")
    assert fake_llm.messages[1][0].startswith("You are a clinical scenario classifier")
    assert fake_llm.messages[2][0].startswith("You are a retrieval route classifier")


def test_invalid_llm_json_falls_back_to_rules() -> None:
    fake_llm = FakeLLMClient(["not-json"])

    plan = QueryPlanner(enable_llm=True, llm_client=fake_llm).plan(
        "Why does aspirin reduce platelet aggregation?",
        ClinicalQueryEnhancer().enhance(
            "Why does aspirin reduce platelet aggregation?"
        ),
    )

    assert plan.scenario == "biomedical_research"
    assert plan.route == "reasoning"


def test_invalid_llm_enum_falls_back_to_rules() -> None:
    fake_llm = FakeLLMClient(
        ['{"complexity": "extreme", "rationale": "invalid label"}']
    )

    plan = QueryPlanner(enable_llm=True, llm_client=fake_llm).plan(
        "What is the relationship between aspirin and platelet aggregation?",
        ClinicalQueryEnhancer().enhance(
            "What is the relationship between aspirin and platelet aggregation?"
        ),
    )

    assert plan.route == "associative"


def test_missing_prompt_config_falls_back_to_rules(tmp_path) -> None:
    fake_llm = FakeLLMClient(['{"complexity": "high"}'])

    plan = QueryPlanner(
        enable_llm=True,
        llm_client=fake_llm,
        prompts_path=tmp_path / "missing-prompts.json",
    ).plan("Why does aspirin reduce platelet aggregation?")

    assert plan.route == "reasoning"


def test_emergency_and_personal_advice_remain_safety_routed() -> None:
    policy = SafetyPolicy()

    emergency = policy.evaluate(
        "I have chest pain and cannot breathe. What should I do?"
    )
    advice = policy.evaluate("What dose of aspirin should I take tonight?")

    assert emergency.action == "refuse"
    assert advice.action == "caution"
