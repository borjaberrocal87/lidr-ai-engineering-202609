import json
from collections.abc import Iterator

from starlette.testclient import TestClient

from app.config import settings
from app.routers import estimations
from app.services.llm_service import (
    EstimationResult,
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
    StreamMetrics,
)


def test_estimate_success(client: TestClient, transcription: str, monkeypatch) -> None:
    def fake_generate_estimation(text: str) -> EstimationResult:
        assert text == transcription
        return EstimationResult(
            estimation="## Estimación: Landing Page\n\n**Total: 150 horas**",
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.2,
            input_tokens=123,
            output_tokens=45,
        )

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.2
    assert body["input_tokens"] == 123
    assert body["output_tokens"] == 45
    assert body["truncated"] is False
    assert "Estimación" in body["estimation"]


def test_estimate_expone_truncado(client: TestClient, transcription: str, monkeypatch) -> None:
    def fake_generate_estimation(text: str) -> EstimationResult:
        return EstimationResult(
            estimation="## Estimación incompleta",
            model="gpt-4o-mini",
            provider="openai",
            truncated=True,
        )

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 200
    assert response.json()["truncated"] is True


def test_estimate_missing_api_key_returns_503(
    client: TestClient, transcription: str, monkeypatch
) -> None:
    def fake_generate_estimation(text: str) -> EstimationResult:
        raise LLMConfigurationError("OPEN_AI_KEY no está configurada.")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 503
    assert "OPEN_AI_KEY" in response.json()["detail"]


def test_estimate_provider_error_returns_502_sin_filtrar_detalle(
    client: TestClient, transcription: str, monkeypatch
) -> None:
    detalle_interno = "sk-secreto-interno-no-debe-salir"

    def fake_generate_estimation(text: str) -> EstimationResult:
        raise LLMProviderError(detalle_interno)

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 502
    assert detalle_interno not in response.text


def test_estimate_input_error_del_servicio_returns_422(
    client: TestClient, transcription: str, monkeypatch
) -> None:
    def fake_generate_estimation(text: str) -> EstimationResult:
        raise LLMInputError("La transcripción no cumple las restricciones.")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 422


def test_estimate_unexpected_error_returns_500(transcription: str, monkeypatch) -> None:
    from app.main import app

    def fake_generate_estimation(text: str) -> EstimationResult:
        raise RuntimeError("bug inesperado")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    failing_client = TestClient(app, raise_server_exceptions=False)
    response = failing_client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 500


def test_estimate_rejects_short_transcription(client: TestClient, monkeypatch) -> None:
    def no_deberia_llamarse(text: str) -> EstimationResult:
        raise AssertionError("El LLM no debe invocarse con entrada inválida")

    monkeypatch.setattr(estimations, "generate_estimation", no_deberia_llamarse)

    response = client.post("/api/v1/estimate", json={"transcription": "corto"})

    assert response.status_code == 422


def test_estimate_rejects_oversized_transcription(client: TestClient, monkeypatch) -> None:
    def no_deberia_llamarse(text: str) -> EstimationResult:
        raise AssertionError("El LLM no debe invocarse con entrada inválida")

    monkeypatch.setattr(estimations, "generate_estimation", no_deberia_llamarse)

    oversized = "x" * (settings.transcription_max_length + 1)
    response = client.post("/api/v1/estimate", json={"transcription": oversized})

    assert response.status_code == 422


def test_estimate_requires_transcription_field(client: TestClient) -> None:
    response = client.post("/api/v1/estimate", json={})

    assert response.status_code == 422


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


def test_estimate_stream_success(client: TestClient, transcription: str, monkeypatch) -> None:
    def fake_stream_estimation(text: str, metrics: StreamMetrics | None = None) -> Iterator[str]:
        assert text == transcription
        if metrics is not None:
            metrics.model = "gpt-4o-mini"
            metrics.provider = "openai"
            metrics.input_tokens = 123
            metrics.output_tokens = 45
        yield "## Estimación"
        yield " completa"

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json={"transcription": transcription})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    assert events[0] == ("token", {"text": "## Estimación"})
    assert events[1] == ("token", {"text": " completa"})
    assert events[2] == (
        "done",
        {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "input_tokens": 123,
            "output_tokens": 45,
            "truncated": False,
        },
    )


def test_estimate_stream_provider_error_emits_error_event(
    client: TestClient, transcription: str, monkeypatch
) -> None:
    detalle_interno = "sk-secreto-interno-no-debe-salir"

    def fake_stream_estimation(text: str, metrics: StreamMetrics | None = None) -> Iterator[str]:
        yield "parcial"
        raise LLMProviderError(detalle_interno)

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json={"transcription": transcription})

    assert response.status_code == 200
    assert detalle_interno not in response.text
    events = _parse_sse(response.text)
    assert events[0] == ("token", {"text": "parcial"})
    assert events[1][0] == "error"
    assert "No se pudo generar" in events[1][1]["detail"]
    assert all(name != "done" for name, _ in events)


def test_estimate_stream_missing_config_emits_error_event(
    client: TestClient, transcription: str, monkeypatch
) -> None:
    def fake_stream_estimation(text: str, metrics: StreamMetrics | None = None) -> Iterator[str]:
        raise LLMConfigurationError("OPEN_AI_KEY no está configurada.")

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json={"transcription": transcription})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0][0] == "error"
    assert "OPEN_AI_KEY" in events[0][1]["detail"]
    assert all(name != "done" for name, _ in events)


def test_estimate_stream_service_input_error_emits_error_event(
    client: TestClient, transcription: str, monkeypatch
) -> None:
    def fake_stream_estimation(text: str, metrics: StreamMetrics | None = None) -> Iterator[str]:
        raise LLMInputError("La transcripción no cumple las restricciones.")

    monkeypatch.setattr(estimations, "stream_estimation", fake_stream_estimation)

    response = client.post("/api/v1/estimate/stream", json={"transcription": transcription})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0][0] == "error"
    assert "restricciones" in events[0][1]["detail"]


def test_estimate_stream_rejects_short_transcription(client: TestClient, monkeypatch) -> None:
    def no_deberia_llamarse(text: str, metrics: StreamMetrics | None = None) -> Iterator[str]:
        raise AssertionError("El LLM no debe invocarse con entrada inválida")

    monkeypatch.setattr(estimations, "stream_estimation", no_deberia_llamarse)

    response = client.post("/api/v1/estimate/stream", json={"transcription": "corto"})

    assert response.status_code == 422


def test_context_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/context")

    assert response.status_code == 200
    body = response.json()
    assert "estimador de software" in body["system_prompt"].lower()
    assert len(body["examples"]) >= 1
    assert body["examples"][0]["meeting_summary"]
    assert body["examples"][0]["estimation"]
    assert body["transcription_min_length"] == settings.transcription_min_length
    assert body["transcription_max_length"] == settings.transcription_max_length
    assert isinstance(body["llm_configured"], bool)
