"""Minimal resolver for operational data paths."""

from __future__ import annotations

from typing import Any

from src.config import Settings


def resolve_ops_facts(cfg: Settings | None = None) -> dict[str, Any]:
    """Return runtime facts needed by healthcheck gates."""
    _ = cfg
    return {
        "paths": {},
    }
