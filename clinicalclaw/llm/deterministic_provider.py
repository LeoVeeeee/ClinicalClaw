"""Deterministic LLM provider baseline used by local experiments."""

from __future__ import annotations


class DeterministicLLMProvider:
    """Deterministic text generator for tests and local experiments."""

    def complete(self, prompt: str) -> str:
        """Return a fixed completion so experiments are reproducible."""

        first_line = (
            prompt.strip().splitlines()[0] if prompt.strip() else "No prompt provided."
        )
        return f"Deterministic response generated from: {first_line}"
