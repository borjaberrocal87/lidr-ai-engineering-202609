"""Endpoints por etapa del wizard RAG."""

from __future__ import annotations

from pydantic import SecretStr
from starlette.testclient import TestClient

from app.config import settings
from app.dependencies import get_semantic_retriever
from app.generation.rag.schemas import (
    Citation,
    EstimationModule,
    EstimationTask,
    GroundedEstimate,
    ProjectBrief,
    RetrievalOutcome,
    RetrievedChunk,
)
from app.main import app

HEADERS = {"X-API-Key": "estimate-secret"}


class FakeWrapper:
    def complete_structured(self, **kwargs):
        model = kwargs["response_model"]
        if model is ProjectBrief:
            return ProjectBrief(functionality="B2B payments", technologies=["python"]), {}
        return _estimate(), {}


class FakeRetriever:
    async def retrieve_chunks(self, **kwargs) -> RetrievalOutcome:
        return RetrievalOutcome(
            chunks=[_chunk()],
            low_confidence=False,
            candidates_evaluated=3,
        )


def _chunk(chunk_id: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        content="[Module: Auth] OAuth",
        sector="finance",
        project_year=2024,
        chunk_type="historical_task",
        distance=0.2,
    )


def _estimate() -> GroundedEstimate:
    return GroundedEstimate(
        total_engineer_days=5,
        modules=[
            EstimationModule(
                name="Auth",
                tasks=[EstimationTask(name="OAuth", engineer_days=5, sources=[1])],
            )
        ],
        duration_weeks=1,
        sources=[Citation(source_id=1, relevance="primary", used_for="auth")],
        assumptions=[],
        confidence="high",
        reasoning="fuente histórica",
    )


def test_brief_stage_returns_brief_and_retrieval_text(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: FakeWrapper())

    response = client.post(
        "/api/v1/estimate/stages/brief",
        json={"transcript": "x" * 120},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["brief"]["functionality"] == "B2B payments"
    assert response.json()["retrieval_text"].startswith("B2B payments")


def test_retrieve_stage_returns_matches(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    app.dependency_overrides[get_semantic_retriever] = lambda: FakeRetriever()
    try:
        response = client.post(
            "/api/v1/estimate/stages/retrieve",
            json={"query_text": "historico de pagos", "top_k": 5},
            headers=HEADERS,
        )
    finally:
        app.dependency_overrides.pop(get_semantic_retriever, None)

    assert response.status_code == 200
    assert response.json()["chunks"][0]["id"] == 1


def test_assemble_stage_reports_dropped_chunks(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))

    response = client.post(
        "/api/v1/estimate/stages/assemble",
        json={
            "chunks": [_chunk(1).model_dump(), _chunk(2).model_dump()],
            "max_context_tokens": 256,
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["token_count"] > 0
    assert response.json()["dropped_count"] >= 0


def test_generate_stage_returns_grounding_signals(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: FakeWrapper())

    response = client.post(
        "/api/v1/estimate/stages/generate",
        json={
            "context_block": '<source id="1">OAuth</source>',
            "brief": ProjectBrief(functionality="B2B payments").model_dump(),
            "kept_chunks": [_chunk(1).model_dump()],
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["fabricated_source_ids"] == []
    assert response.json()["coherent"] is True


def test_structure_stage_returns_a_clean_structure(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: FakeWrapper())

    response = client.post(
        "/api/v1/estimate/stages/structure",
        json={
            "query": {
                "functionality": "Claims portal with OAuth and OCR intake",
                "technologies": ["python", "postgresql"],
                "sector": "finance",
                "scale": "medium",
                "country": "ES",
                "regulations": ["GDPR"],
                "constraints": ["launch before summer"],
            }
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fabricated_source_ids"] == []
    assert body["coherent"] is True
    assert body["estimate"]["modules"][0]["name"] == "Auth"
