"""Tests del logging estructurado del render de prompts (sin fugar contenido)."""

from structlog.testing import capture_logs

from app.prompts.loader import render_estimation_prompt
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


def _rendered_event(logs: list[dict]) -> dict:
    return next(entry for entry in logs if entry["event"] == "prompt.rendered")


def test_render_emits_version_and_hash_metadata() -> None:
    with capture_logs() as logs:
        render_estimation_prompt(_make_request(), version="v1")

    entry = _rendered_event(logs)
    assert entry["use_case"] == "estimation"
    assert entry["version"] == "v1"
    assert entry["system_chars"] > 0
    assert entry["user_chars"] > 0
    assert len(entry["system_hash"]) == 12
    assert len(entry["user_hash"]) == 12
    assert entry["prompt_fingerprint"]
    assert entry["reference_projects"] == 0


def test_render_logs_reference_project_count() -> None:
    request = _make_request(
        reference_projects=[{"name": "CRM seguros", "description": "pólizas y agentes"}]
    )

    with capture_logs() as logs:
        render_estimation_prompt(request)

    assert _rendered_event(logs)["reference_projects"] == 1


def test_render_logs_do_not_leak_prompt_content() -> None:
    request = _make_request(description="MARCADOR-SECRETO-12345 de un proyecto confidencial.")

    with capture_logs() as logs:
        render_estimation_prompt(request)

    serialized = str(logs)
    assert "MARCADOR-SECRETO-12345" not in serialized
    assert "<project_description>" not in serialized
    assert "Tarifa de desarrollo" not in serialized


def test_hash_is_stable_for_same_content() -> None:
    request = _make_request()

    with capture_logs() as first_logs:
        render_estimation_prompt(request)
    with capture_logs() as second_logs:
        render_estimation_prompt(request)

    assert _rendered_event(first_logs)["system_hash"] == _rendered_event(second_logs)["system_hash"]
