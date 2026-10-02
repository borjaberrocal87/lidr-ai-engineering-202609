"""Tests del cliente HTTP del frontend, con `httpx.MockTransport` (sin red)."""

import json
from collections.abc import Callable

import httpx
import pytest

from frontend import client
from frontend.client import (
    ApiConfigurationError,
    ApiError,
    ApiGuardrailError,
    ApiInputError,
    ApiProviderError,
    ApiUnavailableError,
)
from frontend.models import StreamMetrics

Handler = Callable[[httpx.Request], httpx.Response]

PAYLOAD = {
    "description": "Plataforma de inventario para 5 tiendas con alertas y dashboard.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _sse(*events: tuple[str, dict]) -> bytes:
    body = ""
    for name, data in events:
        body += f"event: {name}\ndata: {json.dumps(data)}\n\n"
    return body.encode()


def test_get_context_parses_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/context"
        return httpx.Response(
            200,
            json={
                "system_prompt": "Eres un estimador...",
                "prompt_version": "v2",
                "available_versions": ["v1", "v2"],
                "description_min_length": 20,
                "description_max_length": 50000,
                "llm_configured": True,
            },
        )

    context = client.get_context(client=_client(handler))

    assert context.system_prompt.startswith("Eres")
    assert context.prompt_version == "v2"
    assert context.available_versions == ["v1", "v2"]
    assert context.description_min_length == 20
    assert context.description_max_length == 50000
    assert context.llm_configured is True


def test_get_context_unavailable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión")

    with pytest.raises(ApiUnavailableError):
        client.get_context(client=_client(handler))


def test_get_context_unexpected_status_raises_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    with pytest.raises(ApiError):
        client.get_context(client=_client(handler))


def _structured_body(**overrides) -> dict:
    body = {
        "result": {
            "summary": "Resumen de la estimación.",
            "confidence_pct": 72,
            "phases": [
                {
                    "name": "Discovery",
                    "duration_weeks": 1,
                    "cost_eur": 2000,
                    "summary": "Definición de alcance.",
                }
            ],
            "total_duration_weeks": 1,
            "total_cost_eur": 2000,
        },
        "prompt_version": "v1",
        "cached": False,
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input_tokens": 10,
        "output_tokens": 20,
        "cost_usd": 0.001,
        "fallback_used": False,
    }
    body.update(overrides)
    return body


def test_estimate_posts_typed_payload_and_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/estimate"
        assert json.loads(request.content) == PAYLOAD
        return httpx.Response(200, json=_structured_body())

    result = client.estimate(PAYLOAD, client=_client(handler))

    assert result.result.summary == "Resumen de la estimación."
    assert result.result.confidence_pct == 72
    assert result.result.phases[0].name == "Discovery"
    assert result.result.total_duration_weeks == 1
    assert result.result.total_cost_eur == 2000
    assert result.result.out_of_scope is False
    assert result.prompt_version == "v1"
    assert result.model == "gpt-4o-mini"
    assert result.provider == "openai"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.cost_usd == 0.001
    assert result.cached is False


def test_estimate_marks_out_of_scope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _structured_body(cached=True)
        body["result"]["confidence_pct"] = 10
        body["result"]["summary"] = "Out of scope: falta información de alcance."
        return httpx.Response(200, json=body)

    result = client.estimate(PAYLOAD, client=_client(handler))

    assert result.result.out_of_scope is True
    assert result.cached is True


def test_estimate_sends_prompt_version_query_param() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("prompt_version") == "v2"
        return httpx.Response(200, json=_structured_body(prompt_version="v2"))

    result = client.estimate(PAYLOAD, prompt_version="v2", client=_client(handler))

    assert result.prompt_version == "v2"


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (422, ApiInputError),
        (503, ApiConfigurationError),
        (502, ApiProviderError),
    ],
)
def test_estimate_maps_error_statuses(status_code: int, expected: type[ApiError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "detalle"})

    with pytest.raises(expected):
        client.estimate(PAYLOAD, client=_client(handler))


def test_estimate_unavailable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión")

    with pytest.raises(ApiUnavailableError):
        client.estimate(PAYLOAD, client=_client(handler))


def test_stream_estimation_yields_tokens_and_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/estimate/stream"
        assert json.loads(request.content) == PAYLOAD
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                ("token", {"text": "Hola"}),
                ("token", {"text": " mundo"}),
                (
                    "done",
                    {
                        "prompt_version": "v1",
                        "model": "gpt-4o-mini",
                        "provider": "openai",
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "truncated": False,
                    },
                ),
            ),
        )

    metrics = StreamMetrics()
    chunks = list(client.stream_estimation(PAYLOAD, metrics, client=_client(handler)))

    assert chunks == ["Hola", " mundo"]
    assert metrics.prompt_version == "v1"
    assert metrics.model == "gpt-4o-mini"
    assert metrics.provider == "openai"
    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 2
    assert metrics.truncated is False


def test_stream_estimation_provider_error_event_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                ("token", {"text": "parcial"}),
                ("error", {"detail": "Fallo del proveedor."}),
            ),
        )

    stream = client.stream_estimation(PAYLOAD, client=_client(handler))
    assert next(stream) == "parcial"
    with pytest.raises(ApiProviderError, match="Fallo del proveedor"):
        next(stream)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (422, ApiInputError),
        (503, ApiConfigurationError),
        (502, ApiProviderError),
    ],
)
def test_stream_estimation_maps_error_statuses(status_code: int, expected: type[ApiError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "detalle"})

    with pytest.raises(expected):
        list(client.stream_estimation(PAYLOAD, client=_client(handler)))


def test_stream_estimation_unavailable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión")

    with pytest.raises(ApiUnavailableError):
        list(client.stream_estimation(PAYLOAD, client=_client(handler)))


def test_estimate_maps_guardrail_400() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"detail": {"reason": "pii", "message": "Se detectó un email."}},
        )

    with pytest.raises(ApiGuardrailError) as exc_info:
        client.estimate(PAYLOAD, client=_client(handler))

    assert exc_info.value.reason == "pii"
    assert "email" in exc_info.value.message


def test_stream_guardrail_error_event_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(("error", {"detail": "Se detectó un email.", "reason": "pii"})),
        )

    stream = client.stream_estimation(PAYLOAD, client=_client(handler))
    with pytest.raises(ApiGuardrailError) as exc_info:
        next(stream)

    assert exc_info.value.reason == "pii"
