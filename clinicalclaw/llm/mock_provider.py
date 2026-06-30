"""Mock LLM provider used by the demo."""

from __future__ import annotations


class MockLLMProvider:
    """Deterministic text generator for tests and local demos."""

    def complete(self, prompt: str) -> str:
        """Return a fixed completion so the scaffold is reproducible."""

        first_line = prompt.strip().splitlines()[0] if prompt.strip() else "No prompt provided."
        return f"Mock response generated from: {first_line}"
