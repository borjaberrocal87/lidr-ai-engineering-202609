from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from app.config import DEFAULT_TEMPERATURE, get_settings, settings
from app.dependencies import get_cache, get_llm_wrapper, get_openai_client, get_semantic_cache
from app.main import app


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Evita que las cachés de settings/wrapper/cache arrastren estado entre tests."""
    get_settings.cache_clear()
    get_llm_wrapper.cache_clear()
    get_cache.cache_clear()
    get_openai_client.cache_clear()
    get_semantic_cache.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_wrapper.cache_clear()
    get_cache.cache_clear()
    get_openai_client.cache_clear()
    get_semantic_cache.cache_clear()


@pytest.fixture(autouse=True)
def _base_llm_settings(monkeypatch):
    """Config determinista e independiente del `.env` local.

    En CI no existe `.env` (está gitignoreado), así que fijamos una base
    conocida para que los tests no dependan de credenciales reales. Cada test
    puede sobreescribir lo que necesite después.
    """
    baseline = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_fallback_model": "",
        "llm_routing_mode": "fallback",
        "temperature": DEFAULT_TEMPERATURE,
        "llm_timeout_seconds": 30.0,
        "llm_max_retries": 2,
        "llm_max_tokens": 2048,
        "description_min_length": 20,
        "description_max_length": 50_000,
        "cache_backend": "memory",
        "cache_ttl": 86_400,
        "redis_url": "redis://localhost:6379",
        "guardrails_enabled": True,
        "guardrail_moderation_enabled": False,
        "semantic_cache_enabled": False,
        "open_ai_key": "test-key",
        "anthropic_api_key": "test-key",
        "custom_llm_base_url": "",
        "custom_llm_api_key": "test-key",
    }
    for name, value in baseline.items():
        monkeypatch.setattr(settings, name, value)

    # `app.dependencies` usa `get_settings()` (que releería `.env`); forzamos que
    # devuelva este objeto ya parcheado para que los tests no dependan del `.env`
    # local (p. ej. CACHE_BACKEND=redis) ni de servicios externos.
    monkeypatch.setattr("app.dependencies.get_settings", lambda: settings)


@pytest.fixture(autouse=True)
def _forbid_real_llm(monkeypatch):
    """Ningún test debe llamar a un proveedor real: hay que mockear el wrapper."""

    def _forbidden():
        raise RuntimeError(
            "Llamada al LLM real bloqueada en tests. Mockea "
            "app.services.llm_service.get_llm_wrapper."
        )

    monkeypatch.setattr("app.services.llm_service.get_llm_wrapper", _forbidden)


@pytest.fixture
def client() -> Iterator[TestClient]:
    # `with` ejecuta el lifespan de la app: si algún día se añade uno, los tests
    # lo respetan en vez de saltárselo silenciosamente.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def description() -> str:
    return (
        "En la reunión con el equipo de marketing, el cliente explicó que "
        "necesita una landing page con formulario de contacto, integración con "
        "su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG."
    )


@pytest.fixture
def estimation_payload(description: str) -> dict[str, str]:
    """Payload JSON válido para los endpoints de estimación."""
    return {
        "description": description,
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }


@pytest.fixture
def estimation_request(description: str):
    from app.schemas.estimations import (
        DetailLevel,
        EstimationRequest,
        OutputFormat,
        ProjectType,
    )

    return EstimationRequest(
        description=description,
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )


class FakeWrapper:
    """Doble del `LLMWrapper` para tests del servicio (sin red)."""

    def __init__(
        self,
        *,
        result: dict | None = None,
        chunks: list[str] | None = None,
        error: Exception | None = None,
        structured_result: object | None = None,
        structured_meta: dict | None = None,
    ) -> None:
        self.result = result if result is not None else _default_result()
        self.chunks = chunks if chunks is not None else ["## ", "Estimación"]
        self.error = error
        self.structured_result = (
            structured_result if structured_result is not None else _default_structured_result()
        )
        self.structured_meta = (
            structured_meta if structured_meta is not None else _default_structured_meta()
        )
        self.complete_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.structured_calls: list[dict] = []

    def complete(self, **kwargs) -> dict:
        self.complete_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(self.result)

    def complete_stream(self, **kwargs) -> Iterator[str]:
        self.stream_calls.append(kwargs)
        metrics = kwargs.get("metrics")
        if metrics is not None:
            metrics.model = self.result["model"]
            metrics.provider = self.result["provider"]
            metrics.input_tokens = self.result.get("input_tokens")
            metrics.output_tokens = self.result.get("output_tokens")
            metrics.truncated = bool(self.result.get("truncated", False))
        if self.error is not None:
            raise self.error
        yield from self.chunks

    def complete_structured(self, **kwargs) -> tuple[object, dict]:
        self.structured_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.structured_result, dict(self.structured_meta)


def _default_structured_result() -> object:
    from app.schemas.estimations import EstimationResult, Phase

    return EstimationResult(
        summary="Una landing page con CRM y blog para el equipo de marketing.",
        confidence_pct=70,
        phases=[
            Phase(
                name="Discovery",
                duration_weeks=1,
                cost_eur=2000,
                summary="Entrevistas con marketing y alcance.",
            )
        ],
        total_duration_weeks=1,
        total_cost_eur=2000,
    )


def _default_structured_meta() -> dict:
    return {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input_tokens": 11,
        "output_tokens": 22,
        "cost_usd": 0.000123,
        "latency_ms": 12.5,
        "retries": 0,
        "fallback_used": False,
    }


def _default_result() -> dict:
    return {
        "estimation": "## Estimación de prueba",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input_tokens": 11,
        "output_tokens": 22,
        "truncated": False,
        "latency_ms": 12.5,
        "cost_usd": 0.000123,
        "fallback_used": False,
        "cache_hit": False,
    }
