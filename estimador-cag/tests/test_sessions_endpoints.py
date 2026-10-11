"""Tests de integración de la sesión conversacional (httpx.AsyncClient)."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.config import settings
from app.dependencies import get_session_store
from app.main import app
from app.schemas.critic import CriticFeedback
from app.schemas.estimations import EstimationResult
from app.services import estimation
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionStore

VALID_FORM = {
    "transcript": "We want a CRM called Nimbus built with React for the sales team.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def _meta() -> dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input_tokens": 10,
        "output_tokens": 20,
        "cost_usd": 0.001,
        "fallback_used": False,
    }


def _canned_result() -> EstimationResult:
    return EstimationResult(
        summary="Un CRM B2B de tamaño medio para el equipo de ventas.",
        confidence_pct=72,
        phases=[
            {
                "name": "Descubrimiento",
                "duration_weeks": 1,
                "cost_eur": 5_000,
                "summary": "Talleres y spike técnico del alcance.",
            },
            {
                "name": "Construcción",
                "duration_weeks": 5,
                "cost_eur": 20_000,
                "summary": "Funcionalidad núcleo del CRM.",
            },
        ],
        total_duration_weeks=6,
        total_cost_eur=25_000,
    )


class FakeConversationalWrapper:
    """Doble del wrapper: distingue estimación (messages) de extractor (system/user)."""

    def __init__(
        self,
        estimations: list[EstimationResult] | None = None,
        metadatas: list[ProjectMetadata] | None = None,
    ) -> None:
        self.estimations = estimations or [_canned_result()]
        self.metadatas = metadatas or [ProjectMetadata()]
        self.estimation_calls: list[list[dict[str, str]]] = []
        self.metadata_calls: list[str] = []

    def complete_structured_messages(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        if response_model is CriticFeedback:
            return CriticFeedback(verdict="accept", issues=[], confidence_in_review=90), _meta()
        if response_model is EstimationResult:
            self.estimation_calls.append(messages)
            index = min(len(self.estimation_calls) - 1, len(self.estimations) - 1)
            return self.estimations[index], _meta()
        # Envoltorios auxiliares (resumen acumulativo / ancla vía LLM).
        fields = getattr(response_model, "model_fields", {})
        if "summary" in fields:
            return response_model(summary="Resumen acumulado de turnos previos."), _meta()
        if "is_anchor" in fields:
            return response_model(is_anchor=False), _meta()
        return response_model(), _meta()

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type,
        **kwargs: Any,
    ) -> tuple[ProjectMetadata, dict[str, Any]]:
        self.metadata_calls.append(user_message)
        index = min(len(self.metadata_calls) - 1, len(self.metadatas) - 1)
        return self.metadatas[index], _meta()


@pytest.fixture
def store() -> SessionStore:
    return SessionStore(max_turns=3)


@pytest.fixture
def fake_wrapper() -> FakeConversationalWrapper:
    return FakeConversationalWrapper()


@pytest.fixture
async def client(
    store: SessionStore, fake_wrapper: FakeConversationalWrapper, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr(estimation, "get_llm_wrapper", lambda: fake_wrapper)
    monkeypatch.setattr(settings, "guardrails_enabled", False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


async def _create_session(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/v1/sessions")
    assert response.status_code == 201
    return str(response.json()["session_id"])


async def test_two_turns_accumulate_metadata(
    client: httpx.AsyncClient, fake_wrapper: FakeConversationalWrapper
) -> None:
    fake_wrapper.metadatas = [
        ProjectMetadata(
            project_name="Nimbus",
            assumed_team_size=3,
            mentioned_technologies=["React", "Postgres"],
            agreed_scope="Fase 1 MVP CRM para ventas.",
        ),
        ProjectMetadata(mentioned_technologies=["Stripe"], agreed_scope="Añade facturación."),
    ]
    session_id = await _create_session(client)

    first = await client.post(f"/api/v1/sessions/{session_id}/estimate", data=VALID_FORM)
    assert first.status_code == 200, first.text
    assert first.json()["metadata"]["project_name"] == "Nimbus"

    follow_up = {**VALID_FORM, "transcript": "Now add Stripe-based billing on top of it."}
    second = await client.post(f"/api/v1/sessions/{session_id}/estimate", data=follow_up)
    assert second.status_code == 200, second.text

    info = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    assert info["metadata"]["project_name"] == "Nimbus"
    assert info["metadata"]["assumed_team_size"] == 3
    assert sorted(info["metadata"]["mentioned_technologies"]) == ["Postgres", "React", "Stripe"]
    assert "facturación" in info["metadata"]["agreed_scope"].lower()


async def test_docx_attachment_reaches_the_model(
    client: httpx.AsyncClient, fake_wrapper: FakeConversationalWrapper
) -> None:
    from docx import Document

    document = Document()
    document.add_paragraph("Anexo: Nimbus CRM con React + Postgres.")
    buffer = io.BytesIO()
    document.save(buffer)

    session_id = await _create_session(client)
    response = await client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=VALID_FORM,
        files=[
            (
                "attachments",
                (
                    "spec.docx",
                    buffer.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            )
        ],
    )

    assert response.status_code == 200, response.text
    last_user = next(
        message
        for message in reversed(fake_wrapper.estimation_calls[0])
        if message["role"] == "user"
    )
    assert "--- attachment: spec.docx ---" in last_user["content"]
    assert "Nimbus CRM con React + Postgres." in last_user["content"]


async def test_history_respects_the_sliding_window(
    client: httpx.AsyncClient, fake_wrapper: FakeConversationalWrapper, store: SessionStore
) -> None:
    session_id = await _create_session(client)

    for turn in range(8):
        body = {**VALID_FORM, "transcript": f"Turn {turn}: refine scope details for clarity."}
        response = await client.post(f"/api/v1/sessions/{session_id}/estimate", data=body)
        assert response.status_code == 200, response.text

    # max_turns=3 => system + (resumen?) + 3*2 de ventana + 1 turno actual = 9 máximo.
    for messages in fake_wrapper.estimation_calls:
        assert len(messages) <= 9

    session = store.get_or_404(session_id)
    assert len(session.history.messages) <= 3 * 2


async def test_unknown_session_returns_404(
    client: httpx.AsyncClient, fake_wrapper: FakeConversationalWrapper
) -> None:
    response = await client.post("/api/v1/sessions/does-not-exist/estimate", data=VALID_FORM)
    assert response.status_code == 404
    assert response.json()["detail"] == "session_not_found"


async def test_unsupported_attachment_returns_415(
    client: httpx.AsyncClient, fake_wrapper: FakeConversationalWrapper
) -> None:
    session_id = await _create_session(client)
    response = await client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=VALID_FORM,
        files=[("attachments", ("foto.png", b"\x89PNG\r\n", "image/png"))],
    )
    assert response.status_code == 415
    assert response.json()["detail"]["reason"] == "unsupported_attachment"


async def test_create_session_returns_unique_ids(client: httpx.AsyncClient) -> None:
    first = await _create_session(client)
    second = await _create_session(client)
    assert first != second


async def test_acb_endpoint_returns_audit_trail(client: httpx.AsyncClient) -> None:
    session_id = await _create_session(client)
    response = await client.post(f"/api/v1/sessions/{session_id}/estimate-acb", data=VALID_FORM)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["acb"]["final_decision"] == "accept"
    assert body["acb"]["iterations_run"] == 1
    assert body["result"]["total_cost_eur"] == 25_000
    assert body["session_id"] == session_id
