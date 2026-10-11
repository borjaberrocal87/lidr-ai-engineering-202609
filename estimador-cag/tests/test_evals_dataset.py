"""Tests del harness de evals: carga del golden dataset y contrato de métricas."""

from app.schemas.estimations import EstimationResult, Phase
from evals.dataset import GoldenCase, load_dataset
from evals.metrics import (
    ContentRecallMetric,
    CostBoundsMetric,
    SchemaAdherenceMetric,
    run_all_metrics,
)


def _case(**overrides) -> GoldenCase:
    base = {
        "id": "case-test",
        "transcript": "Queremos un CRM para ventas con React y Postgres.",
        "project_type": "web_saas",
    }
    base.update(overrides)
    return GoldenCase(**base)


def _result(
    *, summary: str = "Un CRM con React y Postgres.", cost: int = 25_000
) -> EstimationResult:
    return EstimationResult(
        summary=summary,
        confidence_pct=72,
        phases=[
            Phase(
                name="Construcción",
                duration_weeks=6,
                cost_eur=cost,
                summary="Funcionalidad núcleo con React y Postgres.",
            )
        ],
        total_duration_weeks=6,
        total_cost_eur=cost,
    )


def test_golden_dataset_loads_and_is_unique() -> None:
    cases = load_dataset()

    assert len(cases) >= 10
    assert all(isinstance(case, GoldenCase) for case in cases)
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))


def test_schema_metric_flags_cost_mismatch() -> None:
    # model_construct bypassa los validadores para simular una regresión.
    broken = EstimationResult.model_construct(
        summary="Resumen de prueba suficientemente largo.",
        confidence_pct=72,
        phases=[Phase(name="A", duration_weeks=1, cost_eur=1000, summary="contenido valido")],
        total_duration_weeks=1,
        total_cost_eur=9999,
    )

    result = SchemaAdherenceMetric().evaluate(_case(), broken)

    assert result.passed is False
    assert "phases sum" in result.details


def test_schema_metric_passes_well_formed_result() -> None:
    result = SchemaAdherenceMetric().evaluate(_case(), _result())

    assert result.passed is True


def test_cost_bounds_passes_out_of_scope_placeholder() -> None:
    case = _case(expected_out_of_scope=True)
    placeholder = EstimationResult(
        summary="Out of scope: falta información para dimensionar el proyecto.",
        confidence_pct=0,
        phases=[
            Phase(
                name="Not estimated",
                duration_weeks=1,
                cost_eur=0,
                summary="No hay información suficiente para dimensionar.",
            )
        ],
        total_duration_weeks=1,
        total_cost_eur=0,
    )

    assert CostBoundsMetric().evaluate(case, placeholder).passed is True


def test_content_recall_detects_missing_technology() -> None:
    case = _case(expected_technologies_any_of=["Snowflake", "dbt"])

    result = ContentRecallMetric().evaluate(case, _result())

    assert result.passed is False
    assert "Snowflake" in result.details


def test_content_recall_skips_out_of_scope_cases() -> None:
    case = _case(expected_out_of_scope=True)

    assert ContentRecallMetric().evaluate(case, _result()).passed is True


def test_run_all_metrics_returns_one_per_metric() -> None:
    metrics = run_all_metrics(_case(), _result())

    assert {m.name for m in metrics} == {"schema_adherence", "cost_bounds", "content_recall"}
