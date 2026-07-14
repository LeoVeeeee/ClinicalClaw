from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest


def test_agentic_optional_dependencies_require_v1_ranges() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    agentic = pyproject["project"]["optional-dependencies"]["agentic"]

    assert "langchain>=1.0,<2.0" in agentic
    assert "langchain-core>=1.0,<2.0" in agentic
    assert "langgraph>=1.0,<2.0" in agentic


def test_installed_langgraph_is_v1_when_available() -> None:
    try:
        installed_version = version("langgraph")
    except PackageNotFoundError:
        pytest.skip("langgraph is not installed")

    assert installed_version.startswith("1.")
