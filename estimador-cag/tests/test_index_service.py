"""Expansión incremental: omite duplicados, aísla fallos y reporta avance."""

from __future__ import annotations

import json
from pathlib import Path

from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.ingest_service import DuplicateDocumentError
from app.generation.rag.schemas import (
    Budget,
    BudgetComponent,
    ClientMetadata,
    IngestResponse,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "task_corpus_contradictions.json"


def _budget(budget_id: str) -> Budget:
    return Budget(
        budget_id=budget_id,
        client_metadata=ClientMetadata(name="ACME", sector="finance", country="ES"),
        project_summary="proyecto de prueba",
        main_technology="python",
        year=2024,
        total_estimated_hours=40,
        components=[
            BudgetComponent(
                component_id="C-1",
                name="API",
                description="API REST",
                estimated_hours=40,
                complexity="medium",
            )
        ],
    )


class FakeIngest:
    """Doble del servicio de ingesta: registra rutas y simula duplicados/fallos."""

    def __init__(self, *, duplicates: set[str] | None = None, failures: set[str] | None = None):
        self.paths: list[str] = []
        self._duplicates = duplicates or set()
        self._failures = failures or set()

    async def ingest(
        self,
        *,
        source_path: str,
        document_type: str,
        budget: Budget,
        chunk_type: str = "budget_component",
    ) -> IngestResponse:
        self.paths.append(source_path)
        if budget.budget_id in self._duplicates:
            raise DuplicateDocumentError(1)
        if budget.budget_id in self._failures:
            raise RuntimeError("documento corrupto")
        return IngestResponse(
            document_id=1,
            chunks_created=2,
            embedding_dimension=1536,
            ingestion_time_ms=1,
        )


async def test_expand_counts_each_outcome_and_reports_progress() -> None:
    ingest = FakeIngest(duplicates={"B-2"}, failures={"B-3"})
    service = CorpusIndexService(ingest)  # type: ignore[arg-type]
    progress: list[int] = []

    result = await service.expand(
        [_budget("B-1"), _budget("B-2"), _budget("B-3")],
        on_progress=progress.append,
    )

    assert result.documents_indexed == 1
    assert result.documents_skipped == 1
    assert result.documents_failed == 1
    assert result.chunks_created == 2
    assert result.documents_processed == 3
    assert progress == [1, 2, 3]
    assert ingest.paths == [
        "corpus-expansion::B-1",
        "corpus-expansion::B-2",
        "corpus-expansion::B-3",
    ]


async def test_expand_uses_the_requested_tags_and_prefix() -> None:
    service = CorpusIndexService(FakeIngest())  # type: ignore[arg-type]

    await service.expand([_budget("B-1")], source_prefix="s11-demo")

    assert service._ingest.paths == ["s11-demo::B-1"]  # type: ignore[attr-defined]


def test_contradictions_fixture_matches_the_budget_contract() -> None:
    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    budgets = [Budget.model_validate(item) for item in corpus]

    assert len(budgets) == 2
    hours_by_task: dict[str, list[int]] = {}
    for budget in budgets:
        for component in budget.components:
            hours_by_task.setdefault(component.name, []).append(component.estimated_hours)
    # La premisa del fixture: mismas tareas, horas muy dispares (40 vs 90).
    assert hours_by_task["Real-time notifications delivery service"] == [40, 90]
