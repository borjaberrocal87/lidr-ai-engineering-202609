"""Métricas binarias para el stress test del CAG (sesión 6, bloque 4).

Viven en ``evals/stress/`` y no en ``evals/metrics.py`` porque operan sobre otra
tupla de entrada: aquí evaluamos ``turn_observed`` y el *snapshot* de la sesión
(``GET /api/v1/sessions/{id}``), no ``(GoldenCase, EstimationResult)``. Mover la
firma de ``run_all_metrics`` para acomodarlas mezclaría dos dominios distintos;
lo que sí compartimos es el tipo :class:`MetricResult`.

Determinismo > sofisticación: nada de embeddings ni LLM-as-judge. Match exacto
case-insensitive sobre los campos textuales del snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evals.metrics import MetricResult

_DEFAULT_DRIFT_WHERE = ("summary", "anchors", "metadata")


def _get(obj: Any, field: str) -> Any:
    """Lee un campo de un ``Mapping`` (payload JSON) o de un objeto (Pydantic)."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(field)
    return getattr(obj, field, None)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LatencyBudgetMetric:
    """1.0 si ``latency_ms <= budget_ms``; 0.0 si no.

    Convierte el SLA en un test: el presupuesto es un contrato de diseño, no una
    bandera que se observa a posteriori.
    """

    name = "latency_budget"

    def __init__(self, budget_ms: int) -> None:
        self.budget_ms = budget_ms

    def evaluate(self, observation: Any) -> MetricResult:
        latency = _as_float(_get(observation, "latency_ms"))
        if latency is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="latency_ms ausente en la observación",
            )
        passed = latency <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"latency_ms={latency:.2f} budget_ms={self.budget_ms}",
        )


class CostBudgetMetric:
    """1.0 si ``cost_usd <= budget_usd``; 0.0 si no."""

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd

    def evaluate(self, observation: Any) -> MetricResult:
        cost = _as_float(_get(observation, "cost_usd"))
        if cost is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="cost_usd ausente en la observación",
            )
        passed = cost <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"cost_usd={cost:.6f} budget_usd={self.budget_usd}",
        )


class MemoryDriftMetric:
    """1.0 si el ``fact`` sobrevive en el snapshot; 0.0 si el CAG lo olvidó.

    Un hecho "sobrevive" si aparece literalmente (case-insensitive) en alguno de
    los campos habilitados en ``where``:

    - ``summary``: el resumen acumulativo rodante.
    - ``anchors``: el contenido de los turnos promovidos a ancla.
    - ``metadata``: ``project_name``, ``agreed_scope`` y ``mentioned_technologies``.

    El snapshot puede ser el payload de ``GET /api/v1/sessions/{id}`` o un objeto
    equivalente (una ``Session`` con ``history``/``metadata`` o un dict).
    """

    name = "memory_drift"

    def __init__(self, fact: str, where: list[str] | None = None) -> None:
        self.fact = fact
        self.where = tuple(where) if where is not None else _DEFAULT_DRIFT_WHERE

    def evaluate(self, session_snapshot: Any) -> MetricResult:
        if not self.fact.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="fact vacío: nada que buscar",
            )
        corpus = self._corpus(session_snapshot)
        passed = self.fact.lower() in corpus
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"fact '{self.fact}' "
                + ("encontrado" if passed else "ausente")
                + f" en {sorted(self.where)}"
            ),
        )

    def _corpus(self, snapshot: Any) -> str:
        parts: list[str] = []
        # Acepta tanto el payload plano (con `history`) como la Session cruda.
        history = _get(snapshot, "history") or snapshot

        if "summary" in self.where:
            parts.append(str(_get(history, "summary") or ""))
        if "anchors" in self.where:
            for anchor in _get(history, "anchors") or []:
                parts.append(str(_get(anchor, "content") or ""))
        if "metadata" in self.where:
            metadata = _get(snapshot, "metadata") or _get(history, "metadata")
            parts.append(str(_get(metadata, "project_name") or ""))
            parts.append(str(_get(metadata, "agreed_scope") or ""))
            for tech in _get(metadata, "mentioned_technologies") or []:
                parts.append(str(tech))
        return "\n".join(parts).lower()
