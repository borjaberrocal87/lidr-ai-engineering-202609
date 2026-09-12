from starlette.testclient import TestClient

from app.routers import estimations
from app.services.llm_service import EstimationResult, LLMConfigurationError


def test_estimate_success(client: TestClient, transcription: str, monkeypatch) -> None:
    def fake_generate_estimation(text: str) -> EstimationResult:
        assert text == transcription
        return EstimationResult(
            estimation="## Estimación: Landing Page\n\n**Total: 150 horas**",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=123,
            output_tokens=45,
        )

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["input_tokens"] == 123
    assert body["output_tokens"] == 45
    assert "Estimación" in body["estimation"]


def test_estimate_missing_api_key_returns_503(
    client: TestClient, transcription: str, monkeypatch
) -> None:
    def fake_generate_estimation(text: str) -> EstimationResult:
        raise LLMConfigurationError("OPEN_AI_KEY no está configurada.")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": transcription})

    assert response.status_code == 503
    assert "OPEN_AI_KEY" in response.json()["detail"]


def test_estimate_rejects_short_transcription(client: TestClient) -> None:
    response = client.post("/api/v1/estimate", json={"transcription": "corto"})

    assert response.status_code == 422


def test_estimate_requires_transcription_field(client: TestClient) -> None:
    response = client.post("/api/v1/estimate", json={})

    assert response.status_code == 422
