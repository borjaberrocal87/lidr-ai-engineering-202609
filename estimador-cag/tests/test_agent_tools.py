"""Tools del agente de estimación: coste determinista, guardrails y dispatch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.generation.agentic.estimation_agent.schemas import SearchBudgetsArgs
from app.generation.agentic.estimation_agent.tools import (
    CONTINGENCY_FACTOR,
    calculate_estimate,
    dispatch_tool,
    search_budgets,
    validate_estimate,
)


async def test_search_budgets_delegates_to_the_backend() -> None:
    seen: list[SearchBudgetsArgs] = []

    async def backend(args: SearchBudgetsArgs) -> list[dict]:
        seen.append(args)
        return [
            {"id": 1, "estimated_hours": 120},
            {"id": 2, "estimated_hours": 180},
        ]

    result = await search_budgets(
        {"query": "SAP integration", "filters": {"sectors": ["logistics"], "component_type": None}},
        backend=backend,
    )

    assert seen and seen[0].query == "SAP integration"
    assert seen[0].filters is not None and seen[0].filters.sectors == ["logistics"]
    assert result["count"] == 2
    assert "hours=[120, 180]" in result["summary"]


async def test_search_budgets_reports_an_empty_result() -> None:
    async def backend(_args: SearchBudgetsArgs) -> list[dict]:
        return []

    result = await search_budgets({"query": "unknown component"}, backend=backend)
    assert result["count"] == 0
    assert "no historical items" in result["summary"]


def test_calculate_estimate_takes_the_median_plus_contingency() -> None:
    result = calculate_estimate(
        {
            "components": [
                {"name": "Auth", "reference_amounts": [100.0, 200.0, 300.0]},
                {"name": "Mobile", "reference_amounts": [80.0]},
            ]
        }
    )

    auth, mobile = result["components"]
    assert auth["estimated_hours"] == round(200 * (1 + CONTINGENCY_FACTOR), 1)
    assert mobile["estimated_hours"] == round(80 * (1 + CONTINGENCY_FACTOR), 1)
    assert auth["unbudgeted"] is False
    assert result["total_hours"] == round(auth["estimated_hours"] + mobile["estimated_hours"], 1)


def test_calculate_estimate_flags_unbudgeted_components_instead_of_inventing_hours() -> None:
    result = calculate_estimate(
        {
            "components": [
                {"name": "Known", "reference_amounts": [100.0]},
                {"name": "Unknown", "reference_amounts": []},
            ]
        }
    )

    unknown = result["components"][1]
    assert unknown["unbudgeted"] is True
    assert unknown["estimated_hours"] == 0.0
    assert result["total_hours"] == result["components"][0]["estimated_hours"]


def test_calculate_estimate_is_deterministic() -> None:
    payload = {"components": [{"name": "Auth", "reference_amounts": [90.0, 120.0]}]}
    assert calculate_estimate(payload) == calculate_estimate(payload)


def test_validate_estimate_passes_a_coherent_estimate() -> None:
    result = validate_estimate(
        {
            "components": [
                {"name": "Auth", "estimated_hours": 115.0, "reference_amounts": [100.0]},
            ],
            "total_hours": 115.0,
        }
    )
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_estimate_flags_a_component_without_references() -> None:
    result = validate_estimate(
        {
            "components": [
                {"name": "Unknown", "estimated_hours": 0.0, "reference_amounts": []},
            ],
            "total_hours": 0.0,
        }
    )
    assert result["ok"] is False
    assert any("no historical reference" in issue for issue in result["issues"])


def test_validate_estimate_flags_hours_outside_the_reference_range() -> None:
    result = validate_estimate(
        {
            "components": [
                {"name": "Auth", "estimated_hours": 1_000.0, "reference_amounts": [100.0]},
            ],
            "total_hours": 1_000.0,
        }
    )
    assert result["ok"] is False
    assert any("outside the plausible range" in issue for issue in result["issues"])


def test_validate_estimate_flags_a_total_that_does_not_match_the_components() -> None:
    result = validate_estimate(
        {
            "components": [
                {"name": "Auth", "estimated_hours": 115.0, "reference_amounts": [100.0]},
            ],
            "total_hours": 50.0,
        }
    )
    assert result["ok"] is False
    assert any("does not match the sum" in issue for issue in result["issues"])


def test_validate_estimate_flags_an_implausible_total() -> None:
    result = validate_estimate(
        {
            "components": [
                {"name": "Runaway", "estimated_hours": 25_000.0, "reference_amounts": [25_000.0]},
            ],
            "total_hours": 25_000.0,
        }
    )
    assert result["ok"] is False
    assert any("implausibly large" in issue for issue in result["issues"])


async def test_dispatch_tool_rejects_an_unknown_tool() -> None:
    async def backend(_args: SearchBudgetsArgs) -> list[dict]:
        return []

    with pytest.raises(ValueError, match="Unknown tool"):
        await dispatch_tool("delete_everything", {}, backend=backend)


async def test_dispatch_tool_surfaces_invalid_arguments_as_validation_errors() -> None:
    async def backend(_args: SearchBudgetsArgs) -> list[dict]:
        return []

    with pytest.raises(ValidationError):
        await dispatch_tool("calculate_estimate", {"components": "not-a-list"}, backend=backend)
