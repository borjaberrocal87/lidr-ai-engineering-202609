"""Tests del endpoint GET/PUT /api/v1/config/models (fakeredis, sin red)."""

from __future__ import annotations

from collections.abc import Iterator

import fakeredis
import pytest
from starlette.testclient import TestClient

from app.config import get_settings, settings
from app.dependencies import get_runtime_config
from app.main import app
from app.services.runtime_config import RuntimeConfigUnavailable, RuntimeModelConfig


@pytest.fixture
def runtime_store() -> Iterator[RuntimeModelConfig]:
    store = RuntimeModelConfig(fakeredis.FakeRedis(decode_responses=True), settings)
    app.dependency_overrides[get_runtime_config] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    yield store
    app.dependency_overrides.pop(get_runtime_config, None)
    app.dependency_overrides.pop(get_settings, None)


def test_get_models_returns_snapshot_and_catalog(client: TestClient, runtime_store) -> None:
    response = client.get("/api/v1/config/models")

    assert response.status_code == 200
    body = response.json()
    assert set(body["models"]) == {
        "PRIMARY_MODEL",
        "FALLBACK_MODEL",
        "CRITIC_MODEL",
        "METADATA_EXTRACTOR_MODEL",
        "COMPRESSION_MODEL",
        "PROPOSITIONAL_CHUNKER_MODEL",
        "CONTEXTUAL_CHUNKER_MODEL",
    }
    assert "gpt-4o-mini" in body["available_models"]
    assert body["embedding_model"] == settings.embedding_model
    assert body["embedding_model_note"]


def test_put_applies_and_resets_override(client: TestClient, runtime_store) -> None:
    response = client.put("/api/v1/config/models", json={"models": {"PRIMARY_MODEL": "gpt-4o"}})

    assert response.status_code == 200
    primary = response.json()["models"]["PRIMARY_MODEL"]
    assert primary == {
        "effective": "gpt-4o",
        "default": settings.llm_model,
        "overridden": True,
    }

    reset = client.put("/api/v1/config/models", json={"models": {"PRIMARY_MODEL": None}})

    assert reset.status_code == 200
    assert reset.json()["models"]["PRIMARY_MODEL"]["overridden"] is False


def test_put_unknown_key_returns_422(client: TestClient, runtime_store) -> None:
    response = client.put("/api/v1/config/models", json={"models": {"NOT_A_MODEL": "gpt-4o"}})

    assert response.status_code == 422


def test_put_model_outside_catalog_returns_422(client: TestClient, runtime_store) -> None:
    response = client.put("/api/v1/config/models", json={"models": {"PRIMARY_MODEL": "gpt-5"}})

    assert response.status_code == 422


def test_put_model_without_provider_key_returns_400(
    client: TestClient, runtime_store, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "open_ai_key", None)

    response = client.put("/api/v1/config/models", json={"models": {"PRIMARY_MODEL": "gpt-4o"}})

    assert response.status_code == 400
    assert "OPEN_AI_KEY" in response.json()["detail"]


def test_put_redis_failure_returns_503(client: TestClient, runtime_store, monkeypatch) -> None:
    def _boom(key: str, value: str | None) -> None:
        raise RuntimeConfigUnavailable("down")

    monkeypatch.setattr(runtime_store, "set", _boom)

    response = client.put("/api/v1/config/models", json={"models": {"PRIMARY_MODEL": "gpt-4o"}})

    assert response.status_code == 503
