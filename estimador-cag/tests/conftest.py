from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_cache, get_llm_wrapper
from app.main import app


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Evita que las cachés de settings/wrapper/cache arrastren estado entre tests."""
    get_settings.cache_clear()
    get_llm_wrapper.cache_clear()
    get_cache.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_wrapper.cache_clear()
    get_cache.cache_clear()


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
def transcription() -> str:
    return (
        "En la reunión con el equipo de marketing, el cliente explicó que "
        "necesita una landing page con formulario de contacto, integración con "
        "su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG."
    )


class FakeWrapper:
    """Doble del `LLMWrapper` para tests del servicio (sin red)."""

    def __init__(
        self,
        *,
        result: dict | None = None,
        chunks: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else _default_result()
        self.chunks = chunks if chunks is not None else ["## ", "Estimación"]
        self.error = error
        self.complete_calls: list[dict] = []
        self.stream_calls: list[dict] = []

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
