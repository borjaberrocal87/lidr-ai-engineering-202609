"""Gate semántico: techo por evidencia + juez que ataja alucinaciones con cita real."""

from __future__ import annotations

from types import SimpleNamespace

from app.generation.rag.quality.hallucination import (
    HallucinationGate,
    combine_line_gate,
    evidence_ceiling_days,
    recorded_hours,
)
from app.generation.rag.schemas import (
    EstimationModule,
    EstimationTask,
    GroundedEstimate,
    LineVerdict,
    RetrievedChunk,
    SourceReference,
)


def _chunk(
    chunk_id: int,
    *,
    content: str = "Component: API\nEstimated hours: 48",
    hours: int | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        content=content,
        chunk_type="budget_component",
        distance=0.2,
        estimated_hours=hours,
    )


def _task(*, days: int | None = 6, cites: tuple[int, ...] = (1,)) -> EstimationTask:
    return EstimationTask(
        name="API",
        engineer_days=days,
        grounded=True,
        citations=[
            SourceReference(
                chunk_id=str(chunk_id),
                document_id="BUDGET-2024-0001",
                evidence="Estimated hours: 48",
            )
            for chunk_id in cites
        ],
    )


def _ungrounded_task() -> EstimationTask:
    return EstimationTask(name="I+D", engineer_days=None, grounded=False)


def _estimate(tasks: list[EstimationTask], module: str = "Backend") -> GroundedEstimate:
    return GroundedEstimate(
        total_engineer_days=sum(task.engineer_days or 0 for task in tasks) or None,
        modules=[EstimationModule(name=module, tasks=tasks)],
        confidence="medium",
        reasoning="derivada de presupuestos históricos",
    )


class _FakeJudge:
    """Wrapper estructurado falso: cuenta llamadas y devuelve dictámenes fijos."""

    def __init__(self, verdicts: list[LineVerdict] | None = None, *, fails: bool = False) -> None:
        self.calls = 0
        self._verdicts = verdicts or []
        self._fails = fails

    def complete_structured(self, **_kwargs: object) -> tuple[object, dict[str, object]]:
        self.calls += 1
        if self._fails:
            raise RuntimeError("proveedor caído")
        return SimpleNamespace(verdicts=self._verdicts), {}


def _gate(judge: _FakeJudge | None) -> HallucinationGate:
    return HallucinationGate(judge=judge, model="fake-model", tolerance=0.5)


def test_recorded_hours_prefers_metadata_then_parses_text() -> None:
    assert recorded_hours(_chunk(1, hours=40)) == 40.0
    assert recorded_hours(_chunk(1, content="integraciones: 12,5 horas")) == 12.5
    assert recorded_hours(_chunk(1, content="sin cifras aquí")) is None


def test_ceiling_sums_only_cited_and_known_chunks() -> None:
    chunks = {str(chunk.id): chunk for chunk in (_chunk(1), _chunk(2, hours=16))}
    task = _task(cites=(1, 2, 99))

    assert evidence_ceiling_days(task, chunks) == (48 + 16) / 8


def test_ceiling_is_none_without_recorded_hours() -> None:
    chunks = {str(chunk.id): chunk for chunk in (_chunk(1, content="prosa", hours=None),)}

    assert evidence_ceiling_days(_task(), chunks) is None


def test_ungrounded_line_is_insufficient() -> None:
    gate = combine_line_gate(_ungrounded_task(), "Backend", None, None, tolerance=0.5)

    assert gate.status == "insufficient"


def test_line_within_ceiling_is_grounded() -> None:
    gate = combine_line_gate(_task(days=6), "Backend", 6.0, None, tolerance=0.5)

    assert gate.status == "grounded"
    assert gate.evidence_days == 6.0
    assert gate.deviation == 0.0


def test_line_far_below_ceiling_is_still_grounded() -> None:
    gate = combine_line_gate(_task(days=2), "Backend", 6.0, None, tolerance=0.5)

    assert gate.status == "grounded"


def test_line_over_ceiling_beyond_tolerance_is_degraded() -> None:
    gate = combine_line_gate(_task(days=12), "Backend", 6.0, None, tolerance=0.5)

    assert gate.status == "degraded"
    assert gate.reason is not None and "12d" in gate.reason


def test_judge_rejection_degrades_a_line_with_real_citation() -> None:
    verdict = LineVerdict(module="Backend", component="API", entailed=False, reason="otra alcance")

    gate = combine_line_gate(_task(days=6), "Backend", 6.0, verdict, tolerance=0.5)

    assert gate.status == "degraded"
    assert gate.reason == "otra alcance"


async def test_audit_without_judge_uses_anchor_only() -> None:
    judge = _FakeJudge()
    report = await _gate(judge).audit(
        _estimate([_task(), _ungrounded_task()]),
        [_chunk(1)],
        use_judge=False,
    )

    assert judge.calls == 0
    assert (report.total_lines, report.grounded_lines, report.insufficient_lines) == (2, 1, 1)
    assert report.has_degraded is False


async def test_audit_degrades_lines_rejected_by_the_judge() -> None:
    verdict = LineVerdict(module="Backend", component="API", entailed=False, reason="inventa")
    judge = _FakeJudge([verdict])

    report = await _gate(judge).audit(_estimate([_task()]), [_chunk(1)])

    assert judge.calls == 1
    assert report.degraded_lines == 1
    assert report.has_degraded is True
    assert report.lines[0].reason == "inventa"


async def test_judge_failure_degrades_gracefully_to_the_anchor() -> None:
    judge = _FakeJudge(fails=True)

    report = await _gate(judge).audit(_estimate([_task(days=6)]), [_chunk(1)])

    assert judge.calls == 1
    assert report.grounded_lines == 1
    assert report.degraded_lines == 0
