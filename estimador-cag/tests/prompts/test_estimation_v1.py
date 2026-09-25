"""Tests del template de prompts v1.

Verifican el contrato del render sin tocar el LLM: que los campos del request
aterrizan en el bloque correcto, que los condicionales solo se activan con su
enum y que `StrictUndefined` detecta variables ausentes. Corren en milisegundos.
"""

import pytest
from jinja2 import Environment, StrictUndefined, TemplateNotFound, UndefinedError

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    ReferenceProject,
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


def test_user_prompt_wraps_description_in_project_description_block() -> None:
    request = _make_request(description="MARCADOR-UNICO-12345 construye una app de agenda.")

    _system, user = render_estimation_prompt(request)

    assert "<project_description>" in user
    assert "</project_description>" in user
    start = user.index("<project_description>")
    end = user.index("</project_description>")
    assert "MARCADOR-UNICO-12345 construye una app de agenda." in user[start:end]


def test_phases_table_keyword_appears_only_when_format_requested() -> None:
    table_system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.PHASES_TABLE)
    )
    narrative_system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.NARRATIVE)
    )

    assert "phases_table" in table_system
    assert "phases_table" not in narrative_system


def test_detailed_includes_assumptions_per_phase_summary_does_not() -> None:
    detailed_system, _ = render_estimation_prompt(_make_request(detail_level=DetailLevel.DETAILED))
    summary_system, _ = render_estimation_prompt(_make_request(detail_level=DetailLevel.SUMMARY))

    assert "enumera las asunciones de cada fase" in detailed_system.lower()
    assert "enumera las asunciones de cada fase" not in summary_system.lower()


def test_examples_block_is_included_in_system_prompt() -> None:
    system, _ = render_estimation_prompt(_make_request())

    assert "<examples>" in system
    assert "</examples>" in system


def test_project_type_is_interpolated_in_user_prompt() -> None:
    _system, user = render_estimation_prompt(_make_request(project_type=ProjectType.DATA_PIPELINE))

    assert "data_pipeline" in user


def test_reference_projects_block_renders_names_when_present() -> None:
    request = _make_request(
        reference_projects=[
            ReferenceProject(
                name="CRM para seguros",
                description="CRM con pólizas, agentes y siniestros.",
                estimation="10 semanas / 32.000 EUR",
            ),
            ReferenceProject(name="Portal de clientes", description="Área privada con facturas."),
        ]
    )

    _system, user = render_estimation_prompt(request)

    assert "<reference_projects>" in user
    assert "</reference_projects>" in user
    assert "CRM para seguros" in user
    assert "Portal de clientes" in user
    assert "10 semanas / 32.000 EUR" in user


def test_reference_projects_block_absent_when_none() -> None:
    _system, user = render_estimation_prompt(_make_request())

    assert "<reference_projects>" not in user


def test_strict_undefined_raises_on_missing_variable() -> None:
    env = Environment(undefined=StrictUndefined, autoescape=True)
    template = env.from_string("Hola {{ variable_inexistente }}")

    with pytest.raises(UndefinedError):
        template.render()


def test_unknown_version_raises() -> None:
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(_make_request(), version="v999")
