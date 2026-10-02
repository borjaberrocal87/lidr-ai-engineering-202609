"""Tests de validación del contrato `EstimationRequest`."""

import pytest
from pydantic import ValidationError

from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
)

VALID_PAYLOAD = {
    "description": "Un SaaS B2B pequeño para gestionar préstamos de material entre equipos.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def test_valid_request_constructs_with_typed_enums() -> None:
    request = EstimationRequest(**VALID_PAYLOAD)

    assert request.description == VALID_PAYLOAD["description"]
    assert request.project_type is ProjectType.WEB_SAAS
    assert request.detail_level is DetailLevel.MEDIUM
    assert request.output_format is OutputFormat.PHASES_TABLE


def test_model_dump_serialises_enums_to_value_strings() -> None:
    request = EstimationRequest(**VALID_PAYLOAD)

    dumped = request.model_dump()

    assert dumped["project_type"] == "web_saas"
    assert dumped["detail_level"] == "medium"
    assert dumped["output_format"] == "phases_table"


def test_description_below_minimum_length_fails() -> None:
    payload = {**VALID_PAYLOAD, "description": "demasiado corta"}

    with pytest.raises(ValidationError) as exc_info:
        EstimationRequest(**payload)

    assert any(err["loc"] == ("description",) for err in exc_info.value.errors())


def test_description_above_maximum_length_fails() -> None:
    payload = {**VALID_PAYLOAD, "description": "x" * 50_001}

    with pytest.raises(ValidationError) as exc_info:
        EstimationRequest(**payload)

    assert any(err["loc"] == ("description",) for err in exc_info.value.errors())


@pytest.mark.parametrize("field", ["project_type", "detail_level", "output_format"])
def test_each_enum_rejects_unknown_values(field: str) -> None:
    payload = {**VALID_PAYLOAD, field: "definitely_not_a_real_value"}

    with pytest.raises(ValidationError) as exc_info:
        EstimationRequest(**payload)

    assert any(err["loc"] == (field,) for err in exc_info.value.errors())


def test_missing_required_enum_fails() -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "project_type"}

    with pytest.raises(ValidationError) as exc_info:
        EstimationRequest(**payload)

    assert any(err["loc"] == ("project_type",) for err in exc_info.value.errors())


def test_reference_projects_is_optional_and_typed() -> None:
    request = EstimationRequest(**VALID_PAYLOAD)
    assert request.reference_projects is None

    with_references = EstimationRequest(
        **VALID_PAYLOAD,
        reference_projects=[
            {"name": "CRM seguros", "description": "Pólizas y agentes", "estimation": "10 semanas"}
        ],
    )
    assert with_references.reference_projects is not None
    assert with_references.reference_projects[0].name == "CRM seguros"
    assert with_references.reference_projects[0].estimation == "10 semanas"


def test_reference_projects_rejects_more_than_five() -> None:
    payload = {
        **VALID_PAYLOAD,
        "reference_projects": [
            {"name": f"proyecto-{i}", "description": "descripción"} for i in range(6)
        ],
    }

    with pytest.raises(ValidationError) as exc_info:
        EstimationRequest(**payload)

    assert any(err["loc"] == ("reference_projects",) for err in exc_info.value.errors())


def _valid_result(**overrides) -> dict:
    base = {
        "summary": "Una landing page con CRM, blog y formulario de contacto.",
        "confidence_pct": 70,
        "phases": [
            {
                "name": "Discovery",
                "duration_weeks": 1,
                "cost_eur": 2000,
                "summary": "Alcance, entrevistas y definición funcional.",
            }
        ],
        "total_duration_weeks": 1,
        "total_cost_eur": 2000,
    }
    base.update(overrides)
    return base


def test_estimation_result_accepts_valid_payload() -> None:
    result = EstimationResult(**_valid_result())

    assert result.total_cost_eur == 2000
    assert result.phases[0].name == "Discovery"
    assert result.out_of_scope is False


def test_estimation_result_rejects_phases_sum_mismatch() -> None:
    with pytest.raises(ValidationError) as exc_info:
        EstimationResult(**_valid_result(total_cost_eur=9999))

    assert "no coincide" in str(exc_info.value)


def test_estimation_result_requires_out_of_scope_prefix_on_low_confidence() -> None:
    with pytest.raises(ValidationError) as exc_info:
        EstimationResult(**_valid_result(confidence_pct=10))

    assert "Out of scope" in str(exc_info.value)


def test_estimation_result_accepts_low_confidence_with_prefix() -> None:
    result = EstimationResult(
        **_valid_result(
            confidence_pct=10,
            summary="Out of scope: falta información de alcance e integraciones.",
        )
    )

    assert result.out_of_scope is True


def test_estimation_result_rejects_empty_phases() -> None:
    with pytest.raises(ValidationError):
        EstimationResult(**_valid_result(phases=[], total_cost_eur=0))


def test_phase_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        Phase(name="X", duration_weeks=0, cost_eur=-1, summary="corta")
