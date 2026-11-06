"""Orquestador RAG: idempotencia, soft-fail, reintento de citas y coherencia."""

from __future__ import annotations

import pytest

from app.generation.rag.errors import IncoherentEstimateError, RagRetrievalError
from app.generation.rag.grounded_estimator import estimate_from_transcript
from app.generation.rag.idempotency import IdempotencyStore
from app.generation.rag.schemas import (
    Citation,
    EstimationModule,
    EstimationTask,
    GroundedEstimate,
    ProjectBrief,
    RetrievalOutcome,
    RetrievedChunk,
)


def _chunk(chunk_id: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        content="[Project: pagos][Component: OAuth]",
        sector="finance",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.2,
    )


def _estimate(*, cites: list[int], confidence: str = "medium") -> GroundedEstimate:
    coherent = confidence != "insufficient"
    return GroundedEstimate(
        total_engineer_days=10 if coherent else None,
        modules=[
            EstimationModule(
                name="Authentication",
                tasks=[EstimationTask(name="OAuth", engineer_days=10, sources=cites)],
            )
        ]
        if coherent
        else [],
        duration_weeks=2 if coherent else None,
        sources=[
            Citation(source_id=source_id, relevance="primary", used_for="auth")
            for source_id in cites
        ],
        assumptions=[],
        confidence=confidence,  # type: ignore[arg-type]
        reasoning="derivada del histórico",
        insufficient_context_explanation=("no hay fuentes comparables" if not coherent else None),
    )


class SequenceWrapper:
    def __init__(
        self,
        estimates: list[GroundedEstimate],
        brief: ProjectBrief | None = None,
    ) -> None:
        self._estimates = list(estimates)
        self._brief = brief or ProjectBrief(functionality="B2B payments platform")
        self.calls: list[dict] = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_model") is ProjectBrief:
            return self._brief, {"model": "gpt-5-mini"}
        if not self._estimates:
            raise RuntimeError("sin respuestas preparadas")
        return self._estimates.pop(0), {"model": "gpt-5"}

    def complete(self, **kwargs):
        return {"estimation": self._brief.functionality}

    @property
    def generation_calls(self) -> list[dict]:
        return [call for call in self.calls if call.get("response_model") is GroundedEstimate]


class FakeRetriever:
    def __init__(self, outcome: RetrievalOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    async def retrieve_chunks(self, **kwargs) -> RetrievalOutcome:
        self.calls.append(kwargs)
        return self.outcome


class CharEncoder:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


@pytest.fixture
def store() -> IdempotencyStore:
    return IdempotencyStore(redis_client=None, ttl=3600)


def _wire(monkeypatch, *, retriever, wrapper, store) -> None:
    monkeypatch.setattr("app.dependencies.get_semantic_retriever", lambda: retriever)
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)
    monkeypatch.setattr("app.dependencies.get_idempotency_store", lambda: store)
    monkeypatch.setattr("app.dependencies.get_token_encoder", lambda: CharEncoder())


async def test_pipeline_produces_a_cited_estimate(monkeypatch, store) -> None:
    retriever = FakeRetriever(
        RetrievalOutcome(chunks=[_chunk(1)], low_confidence=False, candidates_evaluated=5)
    )
    wrapper = SequenceWrapper([_estimate(cites=[1])])
    _wire(monkeypatch, retriever=retriever, wrapper=wrapper, store=store)

    estimate = await estimate_from_transcript("x" * 120)

    assert estimate.total_engineer_days == 10
    assert retriever.calls[0]["top_k"] == 10
    assert len(wrapper.generation_calls) == 1


async def test_low_confidence_short_circuits_to_insufficient(monkeypatch, store) -> None:
    retriever = FakeRetriever(
        RetrievalOutcome(chunks=[], low_confidence=True, candidates_evaluated=3)
    )
    wrapper = SequenceWrapper([])
    _wire(monkeypatch, retriever=retriever, wrapper=wrapper, store=store)

    estimate = await estimate_from_transcript("x" * 120, idempotency_key="low-1")

    assert estimate.confidence == "insufficient"
    assert estimate.total_engineer_days is None
    assert wrapper.generation_calls == []
    assert store.get("low-1") is not None


async def test_idempotency_hit_skips_pipeline(monkeypatch, store) -> None:
    cached = _estimate(cites=[1])
    store.set("cached-1", cached)
    retriever = FakeRetriever(
        RetrievalOutcome(chunks=[], low_confidence=True, candidates_evaluated=0)
    )
    wrapper = SequenceWrapper([])
    _wire(monkeypatch, retriever=retriever, wrapper=wrapper, store=store)

    estimate = await estimate_from_transcript("x" * 120, idempotency_key="cached-1")

    assert estimate == cached
    assert retriever.calls == []
    assert wrapper.calls == []


async def test_fabricated_citations_trigger_one_corrective_retry(monkeypatch, store) -> None:
    retriever = FakeRetriever(
        RetrievalOutcome(chunks=[_chunk(1)], low_confidence=False, candidates_evaluated=5)
    )
    wrapper = SequenceWrapper([_estimate(cites=[99]), _estimate(cites=[1])])
    _wire(monkeypatch, retriever=retriever, wrapper=wrapper, store=store)

    estimate = await estimate_from_transcript("x" * 120)

    assert len(wrapper.generation_calls) == 2
    assert "<correction>" in wrapper.generation_calls[1]["user_message"]
    assert estimate.sources[0].source_id == 1


async def test_unrepaired_citations_downgrade_confidence(monkeypatch, store) -> None:
    retriever = FakeRetriever(
        RetrievalOutcome(chunks=[_chunk(1)], low_confidence=False, candidates_evaluated=5)
    )
    wrapper = SequenceWrapper([_estimate(cites=[99]), _estimate(cites=[99])])
    _wire(monkeypatch, retriever=retriever, wrapper=wrapper, store=store)

    estimate = await estimate_from_transcript("x" * 120)

    assert estimate.confidence == "low"


async def test_incoherent_estimate_is_repaired(monkeypatch, store) -> None:
    retriever = FakeRetriever(
        RetrievalOutcome(chunks=[_chunk(1)], low_confidence=False, candidates_evaluated=5)
    )
    broken = _estimate(cites=[1], confidence="insufficient").model_copy(
        update={"total_engineer_days": 4}
    )
    wrapper = SequenceWrapper([broken, _estimate(cites=[], confidence="insufficient")])
    _wire(monkeypatch, retriever=retriever, wrapper=wrapper, store=store)

    estimate = await estimate_from_transcript("x" * 120)

    assert estimate.confidence == "insufficient"
    assert estimate.total_engineer_days is None


async def test_unrepaired_incoherence_raises(monkeypatch, store) -> None:
    retriever = FakeRetriever(
        RetrievalOutcome(chunks=[_chunk(1)], low_confidence=False, candidates_evaluated=5)
    )
    broken = _estimate(cites=[1], confidence="insufficient").model_copy(
        update={"total_engineer_days": 4}
    )
    wrapper = SequenceWrapper([broken, broken])
    _wire(monkeypatch, retriever=retriever, wrapper=wrapper, store=store)

    with pytest.raises(IncoherentEstimateError):
        await estimate_from_transcript("x" * 120)


async def test_retriever_unavailable_raises(monkeypatch, store) -> None:
    wrapper = SequenceWrapper([])
    _wire(monkeypatch, retriever=None, wrapper=wrapper, store=store)

    with pytest.raises(RagRetrievalError):
        await estimate_from_transcript("x" * 120)
