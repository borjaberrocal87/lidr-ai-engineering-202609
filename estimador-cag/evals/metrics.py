"""Métricas deterministas sobre un par ``(GoldenCase, EstimationResult)``.

El motivo de mantenerlas en el árbol (en lugar de delegar por defecto a jueces
LLM de DeepEval) es que la suite corre sin LLM y nos dice exactamente *por qué*
falló un caso. El ``GEval`` de DeepEval queda cableado como métrica opcional
(``LLMJudge``) que se puede activar si se quiere una segunda opinión cualitativa.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.estimations import OUT_OF_SCOPE_PREFIX, EstimationResult
from evals.dataset import GoldenCase


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    details: str


class SchemaAdherenceMetric:
    """La salida estructurada del actor y sus validadores ya los garantiza
    Instructor; esta métrica existe para detectar si alguna de esas garantías
    *regresa* (p. ej. un cambio futuro elimina un validador). También contrasta
    el número de fases contra el rango esperado, cuando se aporta."""

    name = "schema_adherence"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        phase_sum = sum(p.cost_eur for p in result.phases)
        if phase_sum != result.total_cost_eur:
            problems.append(f"phases sum ({phase_sum}) != total_cost_eur ({result.total_cost_eur})")

        if result.confidence_pct < 30 and not result.summary.startswith(OUT_OF_SCOPE_PREFIX):
            problems.append("low confidence missing 'Out of scope:' prefix")

        if case.expected_phase_count_range:
            lo, hi = case.expected_phase_count_range
            if not (lo <= len(result.phases) <= hi):
                problems.append(f"phase count {len(result.phases)} outside [{lo}, {hi}]")

        score = 1.0 if not problems else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            passed=not problems,
            details="; ".join(problems) or "schema ok",
        )


class CostBoundsMetric:
    """Cota de cordura para coste y duración absolutos. Los casos fuera de
    alcance esperan cost=0/duration=1 (el sobre placeholder)."""

    name = "cost_bounds"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        if case.expected_out_of_scope:
            if result.total_cost_eur != 0 or result.total_duration_weeks != 1:
                problems.append(
                    "expected out-of-scope envelope (cost=0, duration=1), got "
                    f"cost={result.total_cost_eur}, weeks={result.total_duration_weeks}"
                )
        else:
            if case.expected_cost_range_eur:
                lo, hi = case.expected_cost_range_eur
                if not (lo <= result.total_cost_eur <= hi):
                    problems.append(f"cost {result.total_cost_eur} EUR outside [{lo}, {hi}]")
            if case.expected_duration_weeks_range:
                lo, hi = case.expected_duration_weeks_range
                if not (lo <= result.total_duration_weeks <= hi):
                    problems.append(f"duration {result.total_duration_weeks}w outside [{lo}, {hi}]")

        score = 1.0 if not problems else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            passed=not problems,
            details="; ".join(problems) or "cost & duration within bounds",
        )


class ContentRecallMetric:
    """Recall ligero — ¿la prosa del summary / de las fases menciona lo que al
    usuario le importaba de verdad? Los casos fuera de alcance se saltan."""

    name = "content_recall"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        if case.expected_out_of_scope:
            return MetricResult(
                name=self.name,
                score=1.0,
                passed=True,
                details="skipped — case is out-of-scope by design",
            )

        haystack = " ".join(
            [result.summary] + [phase.summary + " " + phase.name for phase in result.phases]
        ).lower()

        missing_summary = [
            term for term in case.expected_in_summary if term.lower() not in haystack
        ]

        tech_hit = True
        if case.expected_technologies_any_of:
            tech_hit = any(t.lower() in haystack for t in case.expected_technologies_any_of)

        problems: list[str] = []
        if missing_summary:
            problems.append(f"missing in summary: {missing_summary}")
        if not tech_hit:
            problems.append(f"none of {case.expected_technologies_any_of} mentioned anywhere")

        score = 1.0 if not problems else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            passed=not problems,
            details="; ".join(problems) or "expected content present",
        )


_DEFAULT_METRICS = (SchemaAdherenceMetric(), CostBoundsMetric(), ContentRecallMetric())


def run_all_metrics(case: GoldenCase, result: EstimationResult) -> list[MetricResult]:
    return [m.evaluate(case, result) for m in _DEFAULT_METRICS]
