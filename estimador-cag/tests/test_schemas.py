"""Tests de validación del contrato `EstimationRequest`."""

import pytest
from pydantic import ValidationError

from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
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
