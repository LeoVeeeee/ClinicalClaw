"""Query planning and retrieval-route selection.

``QueryPlanner`` is the single public planning entry point. It can optionally
use an OpenAI-compatible LLM for the three routing decisions, then falls back
to deterministic local rules whenever LLM planning is unavailable or invalid.
Query complexity remains an internal planning signal and is not exposed on
``QueryPlan``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

from clinicalclaw.models import (
    ClinicalScenario,
    EnhancedQuery,
    QueryComplexity,
    QueryPlan,
    Route,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "routing_rules.json"
)
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
EnableLLM = bool | Literal["auto", "always", "never"]

ALLOWED_ROUTES = set(get_args(Route))
ALLOWED_SCENARIOS = set(get_args(ClinicalScenario))
ALLOWED_COMPLEXITIES = set(get_args(QueryComplexity))

DEFAULT_PROMPTS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "routing_prompts.json"
)

# This is only a last-resort safety net. The complete rule set lives in JSON.
FALLBACK_ROUTING_RULES: dict[str, Any] = {
    "parametric_memory_markers": [],
    "complexity": {"medium_markers": [], "high_markers": []},
    "scenario_order": [],
    "scenario_markers": {},
}


class ChatCompletionClient(Protocol):
    """Minimal interface used by ``QueryPlanner`` for LLM routing."""

    def complete(self, system_message: str, user_message: str) -> str:
        """Return raw assistant text for a system/user message pair."""


class OpenAICompatibleChatClient:
    """Small OpenAI-compatible chat client wrapper loaded only when needed."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = DEFAULT_LLM_MODEL,
        timeout: float = 20.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenAICompatibleChatClient | None":
        """Create a client from DeepSeek/OpenAI-compatible environment variables."""

        _load_optional_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None

        try:
            return cls(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_API_URL"),
                model=model or os.getenv("DEEPSEEK_MODEL", DEFAULT_LLM_MODEL),
                timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "20")),
            )
        except ImportError:
            LOGGER.info(
                "OpenAI package is not installed; using rule-based query planning."
            )
            return None

    def complete(self, system_message: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            stream=False,
        )
        return response.choices[0].message.content or ""


class QueryPlanner:
    """Create a complete query plan with optional LLM routing.

    TODO: replace deterministic fallback rules with a MedBERT or lightweight
    classifier trained on scenario, complexity, and retrieval-route labels.
    """

    def __init__(
        self,
        enable_llm: EnableLLM = "auto",
        llm_client: ChatCompletionClient | None = None,
        model: str | None = None,
        rules_path: str | Path | None = None,
        prompts_path: str | Path | None = None,
    ) -> None:
        self.enable_llm = enable_llm
        self.llm_client = llm_client
        self.model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_LLM_MODEL)
        self.rules = _load_routing_rules(rules_path)
        self.prompts = _load_routing_prompts(prompts_path)
        self._attempted_env_client = False

    def plan(
        self,
        question: str,
        enhanced_query: EnhancedQuery | None = None,
    ) -> QueryPlan:
        """Return a complete query plan for retrieval.

        LLM planning is attempted only when enabled and configured. Any invalid
        response or client error falls back to deterministic local rules.
        """

        cleaned = question.strip()
        retrieval_query = enhanced_query.rewritten_query if enhanced_query else cleaned
        if self._should_try_llm():
            try:
                return self._plan_with_llm(cleaned, retrieval_query)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - fallback is intentional for research demos.
                LOGGER.info(
                    "LLM query planning failed; using rule-based fallback. Error: %s",
                    exc,
                )
        return self._plan_with_rules(cleaned, retrieval_query)

    def _should_try_llm(self) -> bool:
        if self.enable_llm in {False, "never"}:
            return False
        if self.llm_client is not None:
            return True
        if not self._attempted_env_client:
            self._attempted_env_client = True
            self.llm_client = OpenAICompatibleChatClient.from_env(self.model)
        return self.llm_client is not None

    def _plan_with_llm(self, question: str, retrieval_query: str) -> QueryPlan:
        if self.llm_client is None:
            raise RuntimeError("LLM planning requested without an LLM client.")

        complexity_payload = _parse_json_object(
            self.llm_client.complete(
                _get_system_message(self.prompts, "complexity_system_message"),
                _complexity_user_message(question),
            )
        )
        complexity = _validate_label(
            complexity_payload.get("complexity"),
            ALLOWED_COMPLEXITIES,
            "complexity",
        )

        scenario_payload = _parse_json_object(
            self.llm_client.complete(
                _get_system_message(self.prompts, "scenario_system_message"),
                _scenario_user_message(question),
            )
        )
        scenario = _validate_label(
            scenario_payload.get("scenario"),
            ALLOWED_SCENARIOS,
            "scenario",
        )

        route_payload = _parse_json_object(
            self.llm_client.complete(
                _get_system_message(self.prompts, "route_system_message"),
                _route_user_message(question, complexity, scenario),
            )
        )
        route = _validate_label(route_payload.get("route"), ALLOWED_ROUTES, "route")
        subqueries = _build_subqueries(
            retrieval_query,
            scenario,
            route,
            suggested_subqueries=route_payload.get("suggested_subqueries"),
        )
        return QueryPlan(
            original_question=question,
            subqueries=subqueries,
            route=route,
            scenario=scenario,
        )

    def _plan_with_rules(self, question: str, retrieval_query: str) -> QueryPlan:
        complexity = _classify_complexity(question, self.rules)
        scenario = _classify_scenario(question, self.rules)
        route = _select_route(question, complexity, self.rules)
        subqueries = _build_subqueries(retrieval_query, scenario, route)
        return QueryPlan(
            original_question=question,
            subqueries=subqueries,
            route=route,
            scenario=scenario,
        )


def _classify_scenario(
    question: str, rules: dict[str, Any] | None = None
) -> ClinicalScenario:
    active_rules = rules or FALLBACK_ROUTING_RULES
    lowered = question.lower()
    markers_by_scenario = active_rules.get("scenario_markers", {})
    for scenario in active_rules.get("scenario_order", []):
        markers = markers_by_scenario.get(scenario, [])
        if _contains_any(lowered, markers):
            return _validate_label(scenario, ALLOWED_SCENARIOS, "scenario")
    return "general"


def _classify_complexity(
    question: str, rules: dict[str, Any] | None = None
) -> QueryComplexity:
    active_rules = rules or FALLBACK_ROUTING_RULES
    lowered = question.lower()
    complexity_rules = active_rules.get("complexity", {})
    if (
        _contains_any(lowered, complexity_rules.get("medium_markers", []))
        or lowered.count(" and ") >= 1
    ):
        return "medium"
    if _contains_any(lowered, complexity_rules.get("high_markers", [])):
        return "high"
    return "low"


def _select_route(
    question: str,
    complexity: QueryComplexity,
    rules: dict[str, Any] | None = None,
) -> Route:
    active_rules = rules or FALLBACK_ROUTING_RULES
    lowered = question.lower()
    if _contains_any(lowered, active_rules.get("parametric_memory_markers", [])):
        return "parametric_memory"
    if complexity == "high":
        return "reasoning"
    if complexity == "medium":
        return "associative"
    return "atomic"


def _build_subqueries(
    retrieval_query: str,
    scenario: ClinicalScenario,
    route: Route,
    suggested_subqueries: Any = None,
) -> list[str]:
    subqueries = [retrieval_query]
    if isinstance(suggested_subqueries, list):
        subqueries.extend(
            str(item).strip() for item in suggested_subqueries if str(item).strip()
        )
    elif route in {"associative", "reasoning"}:
        subqueries.append(
            f"{scenario.replace('_', ' ')} evidence for {retrieval_query}"
        )

    if route == "reasoning":
        subqueries.append(f"mechanism and clinical rationale for {retrieval_query}")
    return _deduplicate_texts(subqueries)


def _load_routing_rules(rules_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(rules_path) if rules_path is not None else DEFAULT_RULES_PATH
    try:
        with path.open("r", encoding="utf-8") as handle:
            configured_rules = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return FALLBACK_ROUTING_RULES
    if not isinstance(configured_rules, dict):
        return FALLBACK_ROUTING_RULES
    return _merge_rules(FALLBACK_ROUTING_RULES, configured_rules)


def _load_routing_prompts(prompts_path: str | Path | None = None) -> dict[str, Any]:
    """Load LLM system messages from JSON without importing prompt text."""

    path = Path(prompts_path) if prompts_path is not None else DEFAULT_PROMPTS_PATH
    try:
        with path.open("r", encoding="utf-8") as handle:
            prompts = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return prompts if isinstance(prompts, dict) else {}


def _get_system_message(prompts: dict[str, Any], key: str) -> str:
    """Convert a JSON string or list of lines into one system message."""

    value = prompts.get(key)
    if isinstance(value, list):
        message = "\n".join(str(line) for line in value)
    elif isinstance(value, str):
        message = value
    else:
        raise ValueError(f"Missing or invalid routing prompt: {key}")
    if not message.strip():
        raise ValueError(f"Empty routing prompt: {key}")
    return message


def _merge_rules(default: dict[str, Any], configured: dict[str, Any]) -> dict[str, Any]:
    merged = dict(default)
    for key, value in configured.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def _validate_label(value: Any, allowed_values: set[str], field_name: str):
    if value not in allowed_values:
        raise ValueError(
            f"Invalid {field_name}: {value!r}; expected one of {sorted(allowed_values)}"
        )
    return value


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(marker.lower() in text for marker in markers)


def _deduplicate_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value.strip())
    return deduped


def _complexity_user_message(query: str) -> str:
    return f"Classify the complexity of this query:\n\nQuery:\n{query}"


def _scenario_user_message(query: str) -> str:
    return f"Classify the clinical scenario of this query:\n\nQuery:\n{query}"


def _route_user_message(
    query: str, complexity: QueryComplexity, scenario: ClinicalScenario
) -> str:
    return (
        "Decide the retrieval route for this query.\n\n"
        f"Query:\n{query}\n\n"
        f"Complexity:\n{complexity}\n\n"
        f"Clinical scenario:\n{scenario}"
    )


def _load_optional_dotenv() -> None:
    if os.getenv("CLINICALCLAW_DISABLE_DOTENV") == "1":
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parents[2]
    for env_path in (project_root / ".env", project_root / "configs" / ".env"):
        if env_path.exists():
            load_dotenv(env_path)
