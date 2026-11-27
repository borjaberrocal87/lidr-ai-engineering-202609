"""Render a graph run to a plain-text summary (Session 13).

Shared by the HTTP endpoint (``run_text``) and the CLI script so the UI, the
download and the terminal all show the same thing. Mirrors the Session 12
``trace_text`` idea: the server renders it once and the UI just prints/downloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def render_run(state: Mapping[str, Any]) -> str:
    """Render the final graph state as a human-readable run summary."""
    lines = [
        "=" * 78,
        "SESSION 13 — ESTIMATION GRAPH RUN",
        "=" * 78,
        f"estimation_id : {state.get('estimation_id') or '-'}",
        f"status        : {state.get('status') or '-'}",
        "",
        "NODE TIMELINE",
    ]
    node_runs = state.get("node_runs") or []
    if not node_runs:
        lines.append("  (no node runs recorded)")
    for run in node_runs:
        lines.append(f"  - {run.get('name')}: {run.get('duration_ms')} ms")

    lines += ["", "REQUIREMENTS"]
    requirements = state.get("requirements") or []
    if not requirements:
        lines.append("  (none)")
    for requirement in requirements:
        lines.append(f"  - {requirement}")

    lines += ["", "COMPONENTS"]
    components = state.get("components") or []
    if not components:
        lines.append("  (none)")
    for component in components:
        lines.append(f"  - {component.get('name')} [{component.get('category')}]")

    lines += ["", "BUDGET MATCHES (accumulator field)"]
    matches = state.get("budget_matches") or []
    if not matches:
        lines.append("  (none)")
    for match in matches:
        lines.append(
            f"  - {match.get('component')}: {match.get('amount')}h "
            f"(ref {match.get('reference_budget_id')}, distance {match.get('distance')})"
        )
    lines.append(f"  ({len(matches)} matches total)")

    lines += ["", "ESTIMATE"]
    estimate = state.get("estimate") or {}
    estimate_components = estimate.get("components") or []
    if not estimate_components:
        lines.append("  (none)")
    for component in estimate_components:
        hours = component.get("estimated_hours")
        hours_text = "unbudgeted" if component.get("unbudgeted") else f"{hours}h"
        lines.append(
            f"  - {component.get('name')}: {hours_text} — {component.get('rationale', '')}"
        )
    lines.append(f"  TOTAL: {estimate.get('total_hours', 0)}h")

    errors = state.get("errors") or []
    if errors:
        lines += ["", "ERRORS / ISSUES"]
        lines += [f"  - {error}" for error in errors]

    return "\n".join(lines)
