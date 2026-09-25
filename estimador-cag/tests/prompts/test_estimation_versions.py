"""Tests del versionado de prompts: v1 vs v2 y disponibilidad en disco."""

import pytest
from jinja2 import TemplateNotFound

from app.prompts.loader import available_estimation_versions, render_estimation_prompt
from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _make_request(**overrides) -> EstimationRequest:
    base = {
        "description": "Un CRM pequeño para una agencia inmobiliaria: contactos y permisos.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    base.update(overrides)
    return EstimationRequest(**base)


def test_available_versions_include_v1_and_v2() -> None:
    versions = available_estimation_versions()

    assert "v1" in versions
    assert "v2" in versions


def test_v2_differs_from_v1_and_marks_its_tone() -> None:
    v1_system, _ = render_estimation_prompt(_make_request(), version="v1")
    v2_system, _ = render_estimation_prompt(_make_request(), version="v2")

    assert v1_system != v2_system
    assert "revisor escéptico" in v2_system
    assert "revisor escéptico" not in v1_system


def test_v2_includes_its_own_examples() -> None:
    system, _ = render_estimation_prompt(_make_request(), version="v2")

    assert "<examples>" in system
    assert "pipeline de datos" in system.lower()


def test_v2_respects_format_and_detail_conditionals() -> None:
    table_system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.PHASES_TABLE), version="v2"
    )
    narrative_system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.NARRATIVE), version="v2"
    )
    detailed_system, _ = render_estimation_prompt(
        _make_request(detail_level=DetailLevel.DETAILED), version="v2"
    )
    summary_system, _ = render_estimation_prompt(
        _make_request(detail_level=DetailLevel.SUMMARY), version="v2"
    )

    assert "phases_table" in table_system
    assert "phases_table" not in narrative_system
    assert "enumera las asunciones de cada fase" in detailed_system.lower()
    assert "enumera las asunciones de cada fase" not in summary_system.lower()


def test_unknown_version_raises_template_not_found() -> None:
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(_make_request(), version="v999")
