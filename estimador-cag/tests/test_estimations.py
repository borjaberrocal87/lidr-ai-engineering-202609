from starlette.testclient import TestClient

from app.config import settings
from app.routers import estimations
from app.services.llm_service import (
    EstimationResult,
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
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
