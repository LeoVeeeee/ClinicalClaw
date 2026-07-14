"""Clinical query enhancement baselines."""

from __future__ import annotations

import re

from clinicalclaw.models import EnhancedQuery


DEFAULT_TERMINOLOGY_MAP = {
    "flu": "influenza",
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "blood thinner": "anticoagulant",
    "sugar disease": "diabetes mellitus",
}


class ClinicalQueryEnhancer:
    """Rewrite colloquial patient language into retrieval-oriented terms.

    TODO: replace deterministic mappings with MeSH/UMLS-backed terminology
    normalization and a lightweight query rewriting model.
    """

    def __init__(self, terminology_map: dict[str, str] | None = None) -> None:
        self.terminology_map = terminology_map or DEFAULT_TERMINOLOGY_MAP

    def enhance(self, query: str) -> EnhancedQuery:
        """Return a structured pre-retrieval enhancement record."""

        rewritten = query.strip()
        expansion_terms: list[str] = []
        notes: list[str] = []
        for colloquial, medical_term in self.terminology_map.items():
            pattern = re.compile(rf"\b{re.escape(colloquial)}\b", flags=re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(medical_term, rewritten)
                expansion_terms.append(medical_term)
                notes.append(f"Mapped '{colloquial}' to '{medical_term}'.")

        if _is_context_limited(query):
            notes.append(
                "Context-limited consultation query; retrieval should prioritize clinical background evidence."
            )

        if not expansion_terms and rewritten == query.strip():
            notes.append("No terminology rewrite applied.")

        return EnhancedQuery(
            original_query=query,
            rewritten_query=rewritten,
            expansion_terms=expansion_terms,
            notes=notes,
        )


def _is_context_limited(query: str) -> bool:
    lowered = query.lower()
    patient_markers = {"i ", "my ", "me ", "should i", "can i"}
    symptom_markers = {"pain", "fever", "cough", "dizzy", "nausea", "rash"}
    return any(marker in lowered for marker in patient_markers) and any(
        marker in lowered for marker in symptom_markers
    )
