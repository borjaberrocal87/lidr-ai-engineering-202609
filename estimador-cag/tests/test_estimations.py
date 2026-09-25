import json
from collections.abc import Iterator

from starlette.testclient import TestClient

from app.config import settings
from app.routers import estimations
from app.schemas.estimations import EstimationRequest
from app.services.llm_service import (
    EstimationResult,
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
    StreamMetrics,
)


def test_estimate_success(
    client: TestClient,
    estimation_payload: dict[str, str],
    description: str,
    monkeypatch,
) -> None:
    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        assert request.description == description
        assert request.output_format.value == "phases_table"
        return EstimationResult(
            estimation="## Estimación: Landing Page\n\n**Total: 150 horas**",
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.2,
            input_tokens=123,
            output_tokens=45,
            cache_hit=True,
            cost_usd=0.0021,
            fallback_used=True,
        )

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json=estimation_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prompt_version"] == "v1"
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.2
    assert body["input_tokens"] == 123
    assert body["output_tokens"] == 45
    assert body["truncated"] is False
    assert body["cache_hit"] is True
    assert body["cost_usd"] == 0.0021
    assert body["fallback_used"] is True
    assert "Estimación" in body["estimation"]


def test_estimate_expone_truncado(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        return EstimationResult(
            estimation="## Estimación incompleta",
            model="gpt-4o-mini",
            provider="openai",
            truncated=True,
        )

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json=estimation_payload)

    assert response.status_code == 200
    assert response.json()["truncated"] is True


def test_estimate_missing_api_key_returns_503(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        raise LLMConfigurationError("OPEN_AI_KEY no está configurada.")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json=estimation_payload)

    assert response.status_code == 503
    assert "OPEN_AI_KEY" in response.json()["detail"]


def test_estimate_provider_error_returns_502_sin_filtrar_detalle(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    detalle_interno = "sk-secreto-interno-no-debe-salir"

    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        raise LLMProviderError(detalle_interno)

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json=estimation_payload)

    assert response.status_code == 502
    assert detalle_interno not in response.text


def test_estimate_input_error_del_servicio_returns_422(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        raise LLMInputError("La descripción no cumple las restricciones.")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json=estimation_payload)

    assert response.status_code == 422


def test_estimate_unexpected_error_returns_500(
    estimation_payload: dict[str, str], monkeypatch
) -> None:
    from app.main import app

    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        raise RuntimeError("bug inesperado")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    failing_client = TestClient(app, raise_server_exceptions=False)
    response = failing_client.post("/api/v1/estimate", json=estimation_payload)

    assert response.status_code == 500


def test_estimate_rejects_short_description(client: TestClient, monkeypatch) -> None:
    def no_deberia_llamarse(request: EstimationRequest, version: str = "v1") -> EstimationResult:
        raise AssertionError("El LLM no debe invocarse con entrada inválida")

    monkeypatch.setattr(estimations, "generate_estimation", no_deberia_llamarse)

    payload = {
        "description": "corto",
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }
    response = client.post("/api/v1/estimate", json=payload)

    assert response.status_code == 422


def test_estimate_rejects_oversized_description(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    def no_deberia_llamarse(request: EstimationRequest, version: str = "v1") -> EstimationResult:
        raise AssertionError("El LLM no debe invocarse con entrada inválida")

    monkeypatch.setattr(estimations, "generate_estimation", no_deberia_llamarse)

    oversized = {**estimation_payload, "description": "x" * (settings.description_max_length + 1)}
    response = client.post("/api/v1/estimate", json=oversized)

    assert response.status_code == 422


def test_estimate_requires_typed_fields(client: TestClient) -> None:
    response = client.post("/api/v1/estimate", json={})

    assert response.status_code == 422


def test_estimate_rejects_unknown_enum(client: TestClient) -> None:
    payload = {
        "description": "Un CRM pequeño para una agencia inmobiliaria con contactos y permisos.",
        "project_type": "not_a_real_enum",
        "detail_level": "medium",
        "output_format": "phases_table",
    }
    response = client.post("/api/v1/estimate", json=payload)

    assert response.status_code == 422
    assert any(err["loc"][-1] == "project_type" for err in response.json()["detail"])


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name: str | None = None
    data: dict | None = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data = json.loads(line.removeprefix("data: "))
        elif line == "" and event_name is not None:
            events.append((event_name, data or {}))
            event_name, data = None, None
    return events


def test_estimate_stream_success(
    client: TestClient,
    estimation_payload: dict[str, str],
    description: str,
    monkeypatch,
) -> None:
    def fake_stream_estimation(
        request: EstimationRequest,
        metrics: StreamMetrics | None = None,
        version: str = "v1",
    ) -> Iterator[str]:
        assert request.description == description
        if metrics is not None:
            metrics.model = "gpt-4o-mini"
            metrics.provider = "openai"
            metrics.input_tokens = 123
            metrics.output_tokens = 45
        yield "## Estimación"
        yield " completa"

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json=estimation_payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    assert events[0] == ("token", {"text": "## Estimación"})
    assert events[1] == ("token", {"text": " completa"})
    assert events[2] == (
        "done",
        {
            "prompt_version": "v1",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "input_tokens": 123,
            "output_tokens": 45,
            "truncated": False,
            "cache_hit": False,
            "cost_usd": None,
            "fallback_used": False,
        },
    )


def test_estimate_stream_provider_error_emits_error_event(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    detalle_interno = "sk-secreto-interno-no-debe-salir"

    def fake_stream_estimation(
        request: EstimationRequest,
        metrics: StreamMetrics | None = None,
        version: str = "v1",
    ) -> Iterator[str]:
        yield "parcial"
        raise LLMProviderError(detalle_interno)

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json=estimation_payload)

    assert response.status_code == 200
    assert detalle_interno not in response.text
    events = _parse_sse(response.text)
    assert events[0] == ("token", {"text": "parcial"})
    assert events[1][0] == "error"
    assert "No se pudo generar" in events[1][1]["detail"]
    assert all(name != "done" for name, _ in events)


def test_estimate_stream_missing_config_emits_error_event(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    def fake_stream_estimation(
        request: EstimationRequest,
        metrics: StreamMetrics | None = None,
        version: str = "v1",
    ) -> Iterator[str]:
        raise LLMConfigurationError("OPEN_AI_KEY no está configurada.")

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json=estimation_payload)

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0][0] == "error"
    assert "OPEN_AI_KEY" in events[0][1]["detail"]
    assert all(name != "done" for name, _ in events)


def test_estimate_stream_service_input_error_emits_error_event(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    def fake_stream_estimation(
        request: EstimationRequest,
        metrics: StreamMetrics | None = None,
        version: str = "v1",
    ) -> Iterator[str]:
        raise LLMInputError("La descripción no cumple las restricciones.")

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json=estimation_payload)

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0][0] == "error"
    assert "restricciones" in events[0][1]["detail"]


def test_estimate_stream_rejects_short_description(client: TestClient, monkeypatch) -> None:
    def no_deberia_llamarse(
        request: EstimationRequest,
        metrics: StreamMetrics | None = None,
        version: str = "v1",
    ) -> Iterator[str]:
        raise AssertionError("El LLM no debe invocarse con entrada inválida")

    monkeypatch.setattr(estimations, "stream_estimation", no_deberia_llamarse)

    payload = {
        "description": "corto",
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }
    response = client.post("/api/v1/estimate/stream", json=payload)

    assert response.status_code == 422


def test_estimate_accepts_reference_projects(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    seen: dict[str, object] = {}

    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        seen["refs"] = request.reference_projects
        return EstimationResult(estimation="## x", model="gpt-4o-mini", provider="openai")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    payload = {
        **estimation_payload,
        "reference_projects": [{"name": "CRM seguros", "description": "pólizas y agentes"}],
    }
    response = client.post("/api/v1/estimate", json=payload)

    assert response.status_code == 200
    refs = seen["refs"]
    assert isinstance(refs, list)
    assert refs[0].name == "CRM seguros"


def test_context_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/context")

    assert response.status_code == 200
    body = response.json()
    assert "estimador de software" in body["system_prompt"].lower()
    assert "<examples>" in body["system_prompt"]
    assert "examples" not in body
    assert body["prompt_version"] == "v1"
    assert {"v1", "v2"} <= set(body["available_versions"])
    assert body["description_min_length"] == settings.description_min_length
    assert body["description_max_length"] == settings.description_max_length
    assert isinstance(body["llm_configured"], bool)


def test_estimate_accepts_prompt_version_query_param(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    seen: dict[str, str] = {}

    def fake_generate_estimation(
        request: EstimationRequest, version: str = "v1"
    ) -> EstimationResult:
        seen["version"] = version
        return EstimationResult(estimation="## v2", model="gpt-4o-mini", provider="openai")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate?prompt_version=v2", json=estimation_payload)

    assert response.status_code == 200
    assert seen["version"] == "v2"
    assert response.json()["prompt_version"] == "v2"


def test_estimate_unknown_prompt_version_returns_404(
    client: TestClient, estimation_payload: dict[str, str], monkeypatch
) -> None:
    def no_deberia_llamarse(request: EstimationRequest, version: str = "v1") -> EstimationResult:
        raise AssertionError("El LLM no debe invocarse con una versión desconocida")

    monkeypatch.setattr(estimations, "generate_estimation", no_deberia_llamarse)

    response = client.post("/api/v1/estimate?prompt_version=v999", json=estimation_payload)

    assert response.status_code == 404


def test_estimate_stream_unknown_prompt_version_returns_404(
    client: TestClient, estimation_payload: dict[str, str]
) -> None:
    response = client.post("/api/v1/estimate/stream?prompt_version=v999", json=estimation_payload)

    assert response.status_code == 404


def test_context_accepts_prompt_version(client: TestClient) -> None:
    response = client.get("/api/v1/context?prompt_version=v2")

    assert response.status_code == 200
    body = response.json()
    assert body["prompt_version"] == "v2"
    assert "revisor escéptico" in body["system_prompt"]


def test_context_unknown_prompt_version_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/context?prompt_version=v999")

    assert response.status_code == 404
