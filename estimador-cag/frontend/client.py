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
from frontend.models import (
    ContextResponse,
    EstimationResult,
    Phase,
    StreamMetrics,
    StructuredEstimateResponse,
)


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


class ApiGuardrailError(ApiError):
    """La descripción fue rechazada por un guardrail de entrada (400)."""

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason


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


def _guardrail_error(response: httpx.Response) -> ApiGuardrailError:
    """Extrae `{reason, message}` del 400 de guardrail, sin romper si no es JSON."""
    try:
        response.read()
        payload: Any = response.json()
    except Exception:
        return ApiGuardrailError(response.text or "Entrada rechazada por los guardrails.")
    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        return ApiGuardrailError(
            str(detail.get("message", "Entrada rechazada por los guardrails.")),
            reason=str(detail.get("reason", "")),
        )
    return ApiGuardrailError(str(detail))


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    if response.status_code == httpx.codes.BAD_REQUEST:
        raise _guardrail_error(response)
    detail = _detail(response)
    if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY:
        raise ApiInputError(detail)
    if response.status_code == httpx.codes.SERVICE_UNAVAILABLE:
        raise ApiConfigurationError(detail)
    if response.status_code == httpx.codes.BAD_GATEWAY:
        raise ApiProviderError(detail)
    raise ApiError(f"Error {response.status_code} de la API: {detail}")


def get_context(*, client: httpx.Client | None = None) -> ContextResponse:
    """Obtiene el system prompt renderizado y los límites activos."""
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
        prompt_version=payload.get("prompt_version", ""),
        available_versions=list(payload.get("available_versions", [])),
        description_min_length=payload["description_min_length"],
        description_max_length=payload["description_max_length"],
        llm_configured=payload["llm_configured"],
    )


def _version_params(prompt_version: str | None) -> dict[str, str]:
    return {"prompt_version": prompt_version} if prompt_version else {}


def estimate(
    payload: dict[str, Any],
    *,
    prompt_version: str | None = None,
    client: httpx.Client | None = None,
) -> StructuredEstimateResponse:
    """Envía un `EstimationRequest` tipado y devuelve la estimación estructurada."""
    with _acquire_client(client) as http:
        try:
            response = http.post(
                _url("/api/v1/estimate"),
                json=payload,
                params=_version_params(prompt_version),
            )
        except httpx.HTTPError as exc:
            raise ApiUnavailableError(
                f"No se pudo contactar con la API en {config.get_api_base_url()}."
            ) from exc
        _raise_for_status(response)
        body = response.json()

    result_payload = body["result"]
    result = EstimationResult(
        summary=result_payload["summary"],
        confidence_pct=result_payload["confidence_pct"],
        phases=[
            Phase(
                name=phase["name"],
                duration_weeks=phase["duration_weeks"],
                cost_eur=phase["cost_eur"],
                summary=phase["summary"],
            )
            for phase in result_payload["phases"]
        ],
        total_duration_weeks=result_payload["total_duration_weeks"],
        total_cost_eur=result_payload["total_cost_eur"],
    )

    return StructuredEstimateResponse(
        result=result,
        prompt_version=body["prompt_version"],
        cached=bool(body.get("cached", False)),
        model=body.get("model", ""),
        provider=body.get("provider", ""),
        input_tokens=body.get("input_tokens"),
        output_tokens=body.get("output_tokens"),
        cost_usd=body.get("cost_usd"),
        fallback_used=bool(body.get("fallback_used", False)),
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
    payload: dict[str, Any],
    metrics: StreamMetrics | None = None,
    *,
    prompt_version: str | None = None,
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
                json=payload,
                params=_version_params(prompt_version),
            ) as response:
                if response.status_code >= 400:
                    _raise_for_status(response)
                for event_name, data in _iter_sse(response):
                    if event_name == "token":
                        yield str(data.get("text", ""))
                    elif event_name == "done":
                        active.prompt_version = str(data.get("prompt_version", ""))
                        active.model = str(data.get("model", ""))
                        active.provider = str(data.get("provider", ""))
                        active.input_tokens = data.get("input_tokens")
                        active.output_tokens = data.get("output_tokens")
                        active.truncated = bool(data.get("truncated", False))
                        active.cache_hit = bool(data.get("cache_hit", False))
                        active.cost_usd = data.get("cost_usd")
                        active.fallback_used = bool(data.get("fallback_used", False))
                    elif event_name == "error":
                        detail = str(data.get("detail", "Fallo del proveedor LLM."))
                        reason = str(data.get("reason", ""))
                        if reason:
                            raise ApiGuardrailError(detail, reason=reason)
                        raise ApiProviderError(detail)
        except httpx.HTTPError as exc:
            raise ApiUnavailableError(
                f"Se interrumpió la conexión con la API ({config.get_api_base_url()})."
            ) from exc
