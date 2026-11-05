"""Offline, provenance-aware evaluation for long-document datasets.

The default workflow makes no model or network requests. Live inference requires
an explicit CLI opt-in and a case limit.
"""

from .core import build_diagnostics, evaluate, resolve_splits, validate

__all__ = ["build_diagnostics", "evaluate", "resolve_splits", "validate"]
