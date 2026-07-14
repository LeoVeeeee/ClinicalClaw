"""Pytest-wide isolation from local API configuration."""

from __future__ import annotations

import os


# Unit tests must remain deterministic and must never call a user's paid API.
os.environ.setdefault("CLINICALCLAW_DISABLE_DOTENV", "1")
