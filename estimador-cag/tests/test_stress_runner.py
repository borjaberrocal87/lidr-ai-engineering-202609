"""Test de integración del runner de estrés (sesión 6, bloque 5).

Corre la campaña en proceso contra un wrapper LLM falso y comprueba que el CSV
sale con todas las columnas, una fila por turno y las métricas binarias
evaluadas. No toca red.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.schemas.estimations import EstimationResult
from app.services import estimation
from app.sessions.models import ProjectMetadata
from evals.stress import run as stress_run


def _meta() -> dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input_tokens": 10,
        "output_tokens": 20,
        "cost_usd": 0.001,
        "latency_ms": 120.0,
        "fallback_used": False,
    }


def _canned_result() -> EstimationResult:
    return EstimationResult(
        summary="Un CRM llamado Nimbus para el equipo de ventas.",
        confidence_pct=72,
        phases=[
            {
                "name": "Descubrimiento",
                "duration_weeks": 1,
                "cost_eur": 5_000,
                "summary": "Talleres de alcance del CRM Nimbus.",
            }
        ],
        total_duration_weeks=1,
        total_cost_eur=5_000,
    )


class _FakeWrapper:
    def complete_structured_messages(
        self, *, messages: list[dict[str, str]], response_model: type, **kwargs: Any
    ) -> tuple[Any, dict[str, Any]]:
        if response_model is EstimationResult:
            return _canned_result(), _meta()
        fields = getattr(response_model, "model_fields", {})
        if "summary" in fields:
            return response_model(summary="Resumen acumulado del proyecto."), _meta()
        if "is_anchor" in fields:
            return response_model(is_anchor=False), _meta()
        return response_model(), _meta()

    def complete_structured(
        self, *, system_prompt: str, user_message: str, response_model: type, **kwargs: Any
    ) -> tuple[Any, dict[str, Any]]:
        return ProjectMetadata(project_name="Nimbus", mentioned_technologies=["React"]), _meta()


def test_runner_writes_csv_with_all_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(estimation, "get_llm_wrapper", lambda: _FakeWrapper())
    monkeypatch.setattr(settings, "guardrails_enabled", False)
    monkeypatch.setattr(stress_run, "FIXTURES_DIR", tmp_path / "fixtures")

    output = tmp_path / "results.csv"
    exit_code = stress_run.main(
        [
            "--scenarios",
            "growing",
            "--attachment-sizes",
            "0,5",
            "--repeats",
            "1",
            "--growing-turns",
            "3",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # 3 turnos x 2 tamaños x 1 repetición
    assert len(rows) == 6
    assert set(stress_run.CSV_FIELDS).issubset(rows[0].keys())

    # El fact del ancla (Nimbus) sobrevive porque la metadata lo preserva.
    assert all(row["memory_drift"] == "1.0" for row in rows)
    assert all(row["cache_hit_kind"] == "none" for row in rows)
    assert all(row["latency_budget"] == "1.0" for row in rows)

    # El adjunto de 5KB no se cuela en la prosa; el caso sin adjunto no aplica.
    sized = [row for row in rows if row["attachment_kb"] == "5"]
    unsized = [row for row in rows if row["attachment_kb"] == "0"]
    assert all(row["attachment_recall"] == "0.0" for row in sized)
    assert all(row["attachment_recall"] == "" for row in unsized)
    assert all(int(row["attachments_total_chars"]) > 0 for row in sized)
    assert all(int(row["attachments_total_chars"]) == 0 for row in unsized)
