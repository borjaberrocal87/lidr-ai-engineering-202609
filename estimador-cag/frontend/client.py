"""Cliente HTTP de la API de estimaciones.

Único punto por el que el frontend toca el backend. Cualquier UI nueva debería
reutilizarlo en lugar de volver a importar `app.*`.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from frontend import config
from frontend.models import ContextExample, ContextResponse, StreamMetrics


class ApiError(RuntimeError):
    """Error genérico de comunicación con la API."""


class ApiUnavailableError(ApiError):
    """La API no responde (red, timeout, DNS)."""


class ApiInputError(ApiError):
    """La API rechazó la entrada (422)."""


class ApiConfigurationError(ApiError):
    """El proveedor LLM no está configurado en el backend (503)."""


class ApiProviderError(ApiError):
    """El proveedor LLM falló (502 o evento SSE `error`)."""


@contextmanager
def _acquire_client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    """Usa el cliente inyectado (tests) o crea uno propio de vida corta."""
    if client is not None:
        yield client
        return
    with httpx.Client(timeout=config.build_http_timeout()) as owned:
        yield owned


def _url(path: str) -> str:
    return f"{config.get_api_base_url()}{path}"


def _detail(response: httpx.Response) -> str:
    """Extrae el `detail` del cuerpo de error, sin romper si no es JSON."""
    try:
        response.read()
        payload: Any = response.json()
    except Exception:
        return response.text or response.reason_phrase
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    detail = _detail(response)
    if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY:
        raise ApiInputError(detail)
    if response.status_code == httpx.codes.SERVICE_UNAVAILABLE:
        raise ApiConfigurationError(detail)
    if response.status_code == httpx.codes.BAD_GATEWAY:
        raise ApiProviderError(detail)
    raise ApiError(f"Error {response.status_code} de la API: {detail}")


def get_context(*, client: httpx.Client | None = None) -> ContextResponse:
    """Obtiene el system prompt, los ejemplos CAG y los límites activos."""
    with _acquire_client(client) as http:
        try:
            response = http.get(_url("/api/v1/context"))
        except httpx.HTTPError as exc:
            raise ApiUnavailableError(
                f"No se pudo contactar con la API en {config.get_api_base_url()}."
            ) from exc
        _raise_for_status(response)
        payload = response.json()

    return ContextResponse(
        system_prompt=payload["system_prompt"],
        examples=[ContextExample(**example) for example in payload["examples"]],
        transcription_min_length=payload["transcription_min_length"],
        transcription_max_length=payload["transcription_max_length"],
        llm_configured=payload["llm_configured"],
    )


def _iter_sse(response: httpx.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parsea el cuerpo SSE en tuplas `(nombre_evento, datos)`."""
    event_name: str | None = None
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if event_name is not None:
                raw = "\n".join(data_lines)
                yield event_name, json.loads(raw) if raw else {}
            event_name, data_lines = None, []
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())


def stream_estimation(
    transcription: str,
    metrics: StreamMetrics | None = None,
    *,
    client: httpx.Client | None = None,
) -> Iterator[str]:
    """Consume `/api/v1/estimate/stream` y cede el texto token a token.

    Si se pasa `metrics`, se rellena con el evento `done`. Los fallos del
    proveedor a mitad del stream se traducen a `ApiProviderError`.
    """
    active = metrics if metrics is not None else StreamMetrics()
    with _acquire_client(client) as http:
        try:
            with http.stream(
                "POST",
                _url("/api/v1/estimate/stream"),
                json={"transcription": transcription},
            ) as response:
                if response.status_code >= 400:
                    _raise_for_status(response)
                for event_name, data in _iter_sse(response):
                    if event_name == "token":
                        yield str(data.get("text", ""))
                    elif event_name == "done":
                        active.model = str(data.get("model", ""))
                        active.provider = str(data.get("provider", ""))
                        active.input_tokens = data.get("input_tokens")
                        active.output_tokens = data.get("output_tokens")
                        active.truncated = bool(data.get("truncated", False))
                        active.cache_hit = bool(data.get("cache_hit", False))
                        active.cost_usd = data.get("cost_usd")
                        active.fallback_used = bool(data.get("fallback_used", False))
                    elif event_name == "error":
                        raise ApiProviderError(str(data.get("detail", "Fallo del proveedor LLM.")))
        except httpx.HTTPError as exc:
            raise ApiUnavailableError(
                f"Se interrumpió la conexión con la API ({config.get_api_base_url()})."
            ) from exc
