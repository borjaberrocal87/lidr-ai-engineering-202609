"""Runner CLI del stress test del CAG (sesión 6, bloque 5).

Orquesta la campaña ``escenario x tamaño de adjunto x repetición x turno`` y
vuelca ``evals/stress/results.csv`` con una fila por turno. Reusa el patrón de
``evals/run.py``: ``TestClient`` en proceso por defecto, ``httpx`` cuando se pasa
``--http BASE``.

Uso::

    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 \\
        --output evals/stress/results.csv

La observación del turno se lee de ``GET /api/v1/sessions/{id}`` (campo
``last_turn``), no de stdout: es determinista y vale igual en proceso que sobre
HTTP. El evento ``turn_observed`` sigue emitiéndose a structlog para producción.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx
from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions.store import SessionStore
from evals.stress.fixtures.build_pdfs import build_all, marker_for
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from evals.stress.scenarios import Scenario, Turn, select_scenarios

API_PREFIX = "/api/v1"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results.csv"

_OBSERVATION_FIELDS = (
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
)
_CONTEXT_FIELDS = ("scenario", "attachment_kb", "repeat")
_METRIC_FIELDS = ("latency_budget", "cost_budget", "memory_drift", "attachment_recall")
CSV_FIELDS = (*_CONTEXT_FIELDS, *_OBSERVATION_FIELDS, *_METRIC_FIELDS)


class SessionClient(Protocol):
    """Interfaz mínima que ambos transportes (in-process / HTTP) implementan."""

    async def create_session(self) -> str: ...

    async def estimate(
        self,
        session_id: str,
        *,
        scenario: Scenario,
        transcript: str,
        attachment: tuple[str, bytes] | None,
    ) -> dict[str, Any]: ...

    async def get_session(self, session_id: str) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class _InProcessClient:
    """Transporte por ``TestClient`` con un ``SessionStore`` aislado."""

    def __init__(self) -> None:
        self._store = SessionStore()
        app.dependency_overrides[get_session_store] = lambda: self._store
        self._client = TestClient(app)

    @staticmethod
    def _files(attachment: tuple[str, bytes] | None) -> dict[str, Any] | None:
        if attachment is None:
            return None
        name, content = attachment
        return {"attachments": (name, content, "application/pdf")}

    async def create_session(self) -> str:
        response = await asyncio.to_thread(self._client.post, f"{API_PREFIX}/sessions")
        response.raise_for_status()
        return str(response.json()["session_id"])

    async def estimate(
        self,
        session_id: str,
        *,
        scenario: Scenario,
        transcript: str,
        attachment: tuple[str, bytes] | None,
    ) -> dict[str, Any]:
        data = {
            "transcript": transcript,
            "project_type": scenario.project_type,
            "detail_level": scenario.detail_level,
            "output_format": scenario.output_format,
        }
        response = await asyncio.to_thread(
            self._client.post,
            f"{API_PREFIX}/sessions/{session_id}/estimate",
            data=data,
            files=self._files(attachment),
        )
        response.raise_for_status()
        return dict(response.json())

    async def get_session(self, session_id: str) -> dict[str, Any]:
        response = await asyncio.to_thread(
            self._client.get, f"{API_PREFIX}/sessions/{session_id}"
        )
        response.raise_for_status()
        return dict(response.json())

    async def aclose(self) -> None:
        self._client.close()
        app.dependency_overrides.pop(get_session_store, None)


class _HttpClient:
    """Transporte HTTP contra un servidor ya levantado."""

    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=180.0)

    async def create_session(self) -> str:
        response = await self._client.post(f"{API_PREFIX}/sessions")
        response.raise_for_status()
        return str(response.json()["session_id"])

    async def estimate(
        self,
        session_id: str,
        *,
        scenario: Scenario,
        transcript: str,
        attachment: tuple[str, bytes] | None,
    ) -> dict[str, Any]:
        data = {
            "transcript": transcript,
            "project_type": scenario.project_type,
            "detail_level": scenario.detail_level,
            "output_format": scenario.output_format,
        }
        files = None
        if attachment is not None:
            name, content = attachment
            files = {"attachments": (name, content, "application/pdf")}
        response = await self._client.post(
            f"{API_PREFIX}/sessions/{session_id}/estimate", data=data, files=files
        )
        response.raise_for_status()
        return dict(response.json())

    async def get_session(self, session_id: str) -> dict[str, Any]:
        response = await self._client.get(f"{API_PREFIX}/sessions/{session_id}")
        response.raise_for_status()
        return dict(response.json())

    async def aclose(self) -> None:
        await self._client.aclose()


def _attachment_for(kb: int, fixtures: dict[int, Path]) -> tuple[str, bytes] | None:
    if kb == 0:
        return None
    path = fixtures[kb]
    return path.name, path.read_bytes()


def _response_text(payload: dict[str, Any]) -> str:
    """Prosa de la respuesta (summary + resúmenes de fase) en minúsculas."""
    result = payload.get("result") or {}
    parts = [str(result.get("summary", ""))]
    for phase in result.get("phases", []) or []:
        parts.append(str(phase.get("summary", "")))
        parts.append(str(phase.get("name", "")))
    return "\n".join(parts).lower()


def _evaluate_turn(
    *,
    observation: dict[str, Any],
    snapshot: dict[str, Any],
    response_payload: dict[str, Any],
    scenario: Scenario,
    turn: Turn,
    attachment_kb: int,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> dict[str, Any]:
    """Construye la fila del CSV: observación + métricas binarias del turno."""
    latency = LatencyBudgetMetric(latency_budget_ms).evaluate(observation)
    cost = CostBudgetMetric(cost_budget_usd).evaluate(observation)

    if turn.turn_index >= scenario.anchor_turn:
        drift = MemoryDriftMetric(scenario.anchor_fact).evaluate(snapshot)
        memory_drift: float | str = drift.score
    else:
        memory_drift = ""  # el hecho del ancla aún no se ha declarado

    if attachment_kb > 0:
        marker = marker_for(attachment_kb).lower()
        attachment_recall: float | str = 1.0 if marker in _response_text(response_payload) else 0.0
    else:
        attachment_recall = ""

    row: dict[str, Any] = {
        "scenario": scenario.name,
        "attachment_kb": attachment_kb,
        "repeat": None,  # lo rellena el llamante
        "latency_budget": latency.score,
        "cost_budget": cost.score,
        "memory_drift": memory_drift,
        "attachment_recall": attachment_recall,
    }
    for field in _OBSERVATION_FIELDS:
        row[field] = observation.get(field)
    return row


async def _run_campaign(
    client: SessionClient,
    *,
    scenarios: Sequence[Scenario],
    attachment_sizes: Sequence[int],
    repeats: int,
    fixtures: dict[int, Path],
    latency_budget_ms: int,
    cost_budget_usd: float,
    errors: list[str],
    on_row: Callable[[dict[str, Any]], None],
) -> int:
    """Recorre la matriz y entrega cada fila a ``on_row`` según se produce.

    Escribir de forma incremental (en vez de acumular y volcar al final) permite
    seguir el progreso de una campaña larga y no perder lo medido si se corta.
    """
    row_count = 0
    for scenario in scenarios:
        for size_kb in attachment_sizes:
            attachment = _attachment_for(size_kb, fixtures)
            for repeat in range(repeats):
                session_id = await client.create_session()
                for turn in scenario.turns:
                    try:
                        payload = await client.estimate(
                            session_id,
                            scenario=scenario,
                            transcript=turn.transcript,
                            attachment=attachment,
                        )
                        snapshot = await client.get_session(session_id)
                    except Exception as exc:  # una fila rota no debe tumbar la campaña
                        message = (
                            f"{scenario.name}/{size_kb}KB/rep{repeat}/turn{turn.turn_index}: "
                            f"{type(exc).__name__}: {str(exc)[:160]}"
                        )
                        errors.append(message)
                        print(f"  ERROR {message}", flush=True)
                        continue
                    observation = snapshot.get("last_turn") or {}
                    row = _evaluate_turn(
                        observation=observation,
                        snapshot=snapshot,
                        response_payload=payload,
                        scenario=scenario,
                        turn=turn,
                        attachment_kb=size_kb,
                        latency_budget_ms=latency_budget_ms,
                        cost_budget_usd=cost_budget_usd,
                    )
                    row["repeat"] = repeat
                    on_row(row)
                    row_count += 1
    return row_count


class _CsvSink:
    """Escritor CSV incremental: cabecera al abrir y ``flush`` por fila."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=list(CSV_FIELDS), extrasaction="ignore"
        )
        self._writer.writeheader()
        self._handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


async def _main_async(args: argparse.Namespace) -> int:
    scenarios = select_scenarios(args.scenarios, growing_turns=args.growing_turns)
    attachment_sizes = args.attachment_sizes

    fixtures: dict[int, Path] = {}
    sizes_to_build = tuple(sorted({kb for kb in attachment_sizes if kb > 0}))
    if sizes_to_build:
        fixtures = build_all(out_dir=FIXTURES_DIR, sizes=sizes_to_build)

    if args.http:
        client: SessionClient = _HttpClient(args.http)
        print(f"Running stress campaign against {args.http}")
    else:
        client = _InProcessClient()
        print("Running stress campaign in-process (TestClient)")

    errors: list[str] = []
    sink = _CsvSink(args.output)
    try:
        row_count = await _run_campaign(
            client,
            scenarios=scenarios,
            attachment_sizes=attachment_sizes,
            repeats=args.repeats,
            fixtures=fixtures,
            latency_budget_ms=args.latency_budget_ms,
            cost_budget_usd=args.cost_budget_usd,
            errors=errors,
            on_row=sink.write,
        )
    finally:
        await client.aclose()
        sink.close()

    print(f"\n{row_count} filas escritas en {args.output}")
    if errors:
        print(f"{len(errors)} turnos fallaron (ver arriba).")
    return 0 if row_count else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", default=None, help="Base URL (p. ej. http://localhost:8000).")
    parser.add_argument(
        "--scenarios",
        default="growing,pivot,contradiction",
        help="Lista separada por comas: growing,pivot,contradiction.",
    )
    parser.add_argument(
        "--attachment-sizes",
        default="0,5,20,50,100",
        help="Tamaños de adjunto en KB (0 = sin adjunto).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeticiones por celda.")
    parser.add_argument("--growing-turns", type=int, default=20, help="Turnos del perfil growing.")
    parser.add_argument("--latency-budget-ms", type=int, default=4000, help="SLA de latencia.")
    parser.add_argument(
        "--cost-budget-usd", type=float, default=0.05, help="Presupuesto por turno."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.scenarios = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    args.attachment_sizes = [
        int(value) for value in args.attachment_sizes.split(",") if value.strip()
    ]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_main_async(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
