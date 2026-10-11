"""Tests de las métricas del stress test (sesión 6, bloque 4)."""

from __future__ import annotations

from types import SimpleNamespace

from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)


def _observation(*, latency_ms: float | None, cost_usd: float | None) -> SimpleNamespace:
    return SimpleNamespace(latency_ms=latency_ms, cost_usd=cost_usd)


# --- LatencyBudgetMetric -----------------------------------------------------


def test_latency_budget_passes_under_budget() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate(
        _observation(latency_ms=1200.0, cost_usd=0.0)
    )
    assert result.passed is True
    assert result.score == 1.0


def test_latency_budget_fails_over_budget() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate(
        _observation(latency_ms=4200.5, cost_usd=0.0)
    )
    assert result.passed is False
    assert result.score == 0.0


def test_latency_budget_boundary_is_inclusive() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate(
        _observation(latency_ms=4000.0, cost_usd=0.0)
    )
    assert result.passed is True


def test_latency_budget_without_metric_fails() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate(
        _observation(latency_ms=None, cost_usd=0.0)
    )
    assert result.passed is False
    assert "ausente" in result.details


# --- CostBudgetMetric --------------------------------------------------------


def test_cost_budget_passes_under_budget() -> None:
    result = CostBudgetMetric(budget_usd=0.05).evaluate(
        _observation(latency_ms=0.0, cost_usd=0.012)
    )
    assert result.passed is True


def test_cost_budget_fails_over_budget() -> None:
    result = CostBudgetMetric(budget_usd=0.01).evaluate(_observation(latency_ms=0.0, cost_usd=0.02))
    assert result.passed is False
    assert result.score == 0.0


def test_cost_budget_reads_dict_payload() -> None:
    result = CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.005})
    assert result.passed is True


# --- MemoryDriftMetric -------------------------------------------------------


def test_memory_drift_found_in_summary() -> None:
    snapshot = {
        "summary": "The project Nimbus now includes billing.",
        "anchors": [],
        "metadata": {},
    }
    result = MemoryDriftMetric("Nimbus").evaluate(snapshot)
    assert result.passed is True


def test_memory_drift_found_in_metadata_project_name() -> None:
    snapshot = {"summary": None, "anchors": [], "metadata": {"project_name": "Nimbus"}}
    result = MemoryDriftMetric("Nimbus").evaluate(snapshot)
    assert result.passed is True


def test_memory_drift_found_in_anchors_content() -> None:
    snapshot = {
        "summary": None,
        "anchors": [{"role": "user", "content": "budget locked at 30000 EUR"}],
        "metadata": {},
    }
    result = MemoryDriftMetric("30000 EUR").evaluate(snapshot)
    assert result.passed is True


def test_memory_drift_is_case_insensitive() -> None:
    snapshot = {"summary": None, "anchors": [], "metadata": {"project_name": "NIMBUS"}}
    result = MemoryDriftMetric("nimbus").evaluate(snapshot)
    assert result.passed is True


def test_memory_drift_lost_returns_zero() -> None:
    snapshot = {"summary": "Un CRM sin nombre.", "anchors": [], "metadata": {}}
    result = MemoryDriftMetric("Nimbus").evaluate(snapshot)
    assert result.passed is False
    assert result.score == 0.0


def test_memory_drift_respects_where_filter() -> None:
    snapshot = {
        "summary": None,
        "anchors": [{"role": "user", "content": "Nimbus"}],
        "metadata": {},
    }
    result = MemoryDriftMetric("Nimbus", where=["summary"]).evaluate(snapshot)
    assert result.passed is False


def test_memory_drift_reads_session_like_object() -> None:
    session = SimpleNamespace(
        history=SimpleNamespace(
            summary="El proyecto Nimbus sigue vivo.",
            anchors=[],
        ),
        metadata=SimpleNamespace(
            project_name="Nimbus",
            agreed_scope=None,
            mentioned_technologies=["React"],
        ),
    )
    result = MemoryDriftMetric("Nimbus").evaluate(session)
    assert result.passed is True
