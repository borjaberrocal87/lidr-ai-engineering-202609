"""Servicio de estimación: orquestador único de las pipelines.

Este módulo absorbe la orquestación que antes vivía en ``llm_service.py`` y la
organiza en una clase ``EstimationService`` (espejo del repositorio de
referencia). La clase agrupa los tres recorridos del dominio:

- ``estimate``: estimación estructurada de un formulario (una sola llamada).
- ``estimate_conversational``: turno multi-turno con memoria (sesión 5).
- ``estimate_with_acb``: variante Actor-Critic-Boss (sesión 5, directo).

El camino de **texto libre / streaming** (legado de la sesión 3) vive en
``app/services/streaming.py``: es un contrato distinto (texto token a token) y
no forma parte del orquestador estructurado.

Las dependencias de infraestructura (wrapper LLM, caché exacta y semántica,
cliente OpenAI) se resuelven a través de ``app.dependencies`` y se invocan en el
momento de la llamada, de modo que los tests puedan sustituirlas sin instanciar
el servicio con dobles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import structlog

from app.config import DEFAULT_TEMPERATURE, reveal_secret, settings
from app.dependencies import (
    get_cache,
    get_llm_wrapper,
    get_openai_client,
    get_semantic_cache,
)
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.prompts.loader import (
    DEFAULT_CONVERSATIONAL_PROMPT_VERSION,
    DEFAULT_ESTIMATION_PROMPT_VERSION,
    prompt_fingerprint,
    render_conversational_prompt,
    render_estimation_prompt,
)
from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.services.cache import make_cache_key
from app.services.errors import (
    LLMConfigurationError as LLMConfigurationError,
)
from app.services.errors import (
    LLMInputError as LLMInputError,
)
from app.services.errors import (
    LLMProviderError as LLMProviderError,
)
from app.sessions.metadata_extractor import update_metadata
from app.sessions.models import Session

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@dataclass
class StructuredEstimation:
    """Resultado estructurado y validado (sesión 4)."""

    result: EstimationResult
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    fallback_used: bool = False
    cache_hit: bool = False


@dataclass
class SessionEstimation:
    """Resultado de un turno conversacional (sesión 5). Nunca viene de caché."""

    result: EstimationResult
    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    fallback_used: bool = False


# --- Helpers compartidos con el camino de streaming -------------------------


def check_description(description: str) -> None:
    """Guard de longitud en el servicio, por si el llamante no es la API.

    La API y Streamlit ya validan en el borde; esto evita que un worker, un CLI
    o una cola manden texto sin techo al proveedor.
    """
    min_length = settings.description_min_length
    max_length = settings.description_max_length
    if not min_length <= len(description) <= max_length:
        raise LLMInputError(
            f"La descripción debe tener entre {min_length} y {max_length} caracteres "
            f"(tiene {len(description)})."
        )


def require_configuration() -> None:
    """Valida de forma anticipada que el proveedor activo tiene credenciales."""
    provider = settings.llm_provider
    if provider == "openai":
        if not reveal_secret(settings.open_ai_key):
            raise LLMConfigurationError("OPEN_AI_KEY no está configurada. Añádela al archivo .env.")
    elif provider == "anthropic":
        if not reveal_secret(settings.anthropic_api_key):
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY no está configurada. Añádela al archivo .env."
            )
    elif provider == "custom":
        if not settings.custom_llm_base_url:
            raise LLMConfigurationError(
                "CUSTOM_LLM_BASE_URL no está configurada. Añádela al archivo .env."
            )
        if not reveal_secret(settings.custom_llm_api_key):
            raise LLMConfigurationError(
                "CUSTOM_LLM_API_KEY no está configurada. Añádela al archivo .env."
            )
    else:
        raise LLMConfigurationError(
            f"Proveedor LLM no soportado: {provider!r}. "
            "Usa 'openai', 'anthropic' o 'custom' en LLM_PROVIDER."
        )


_temperature_warning_emitted = False


def warn_if_temperature_ignored() -> None:
    """Avisa una sola vez si se configura una temperatura que Anthropic ignora.

    El SDK de Anthropic (1.5.0) no acepta el parámetro `temperature`, así que la
    variable `TEMPERATURE` es configuración muerta para ese proveedor.
    """
    global _temperature_warning_emitted
    if _temperature_warning_emitted:
        return
    if settings.llm_provider == "anthropic" and settings.temperature != DEFAULT_TEMPERATURE:
        logger.warning(
            "llm.temperature.ignored",
            provider="anthropic",
            temperature=settings.temperature,
        )
        _temperature_warning_emitted = True


def temperature_for_result() -> float | None:
    """Temperatura efectiva: Anthropic no la admite, el resto sí."""
    if settings.llm_provider == "anthropic":
        return None
    return settings.temperature


def cache_key_material(request: EstimationRequest, version: str) -> str:
    """Material canónico de la clave de caché, propiedad del dominio.

    Incluye el request completo, la versión del prompt y la huella de sus
    templates. Así la caché no depende del texto renderizado y cualquier cambio
    relevante (request o edición de un `.j2`) produce una clave distinta.
    """
    return json.dumps(
        {
            "use_case": "estimation",
            "prompt_version": version,
            "prompt_fingerprint": prompt_fingerprint(version),
            "request": request.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def run_input_guardrails(description: str) -> None:
    """Ejecuta los guardrails de entrada si están activos.

    La moderación solo se invoca si hay cliente (clave OpenAI y flag activo); las
    capas regex corren siempre.
    """
    if not settings.guardrails_enabled:
        return
    client = get_openai_client() if settings.has_moderation else None
    check_input(description, openai_client=client)


class EstimationService:
    """Punto de entrada único de la pipeline estructurada.

    Los knobs de configuración se leen de ``settings`` en el momento de cada
    llamada (no se congelan en el constructor) para que los tests puedan
    sobreescribir la configuración sin reinstanciar el servicio.
    """

    def __init__(
        self,
        *,
        prompt_version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
        conversational_prompt_version: str = DEFAULT_CONVERSATIONAL_PROMPT_VERSION,
        metadata_extractor_model: str | None = None,
    ) -> None:
        self.prompt_version = prompt_version
        self.conversational_prompt_version = conversational_prompt_version
        self.metadata_extractor_model = metadata_extractor_model

    # -- Estimación estructurada (sesión 4) ----------------------------------

    def estimate(
        self,
        request: EstimationRequest,
        *,
        version: str | None = None,
    ) -> StructuredEstimation:
        """Genera una estimación **estructurada** validada, con caché exact-match.

        La clave de caché la construye el dominio (request + versión + huella del
        prompt) y la capa de caché añade los knobs de generación. Solo se cachea el
        resultado ya validado por Pydantic.
        """
        active_version = version or self.prompt_version
        check_description(request.description)
        require_configuration()
        run_input_guardrails(request.description)
        warn_if_temperature_ignored()

        system_prompt, user_message = render_estimation_prompt(request, version=active_version)
        cache_key = make_cache_key(
            cache_key=cache_key_material(request, active_version),
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.temperature,
        )

        cache = get_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return StructuredEstimation(
                result=EstimationResult.model_validate(cached["result"]),
                model=str(cached.get("model", "")),
                provider=str(cached.get("provider", "")),
                input_tokens=cached.get("input_tokens"),
                output_tokens=cached.get("output_tokens"),
                cost_usd=cached.get("cost_usd"),
                fallback_used=bool(cached.get("fallback_used", False)),
                cache_hit=True,
            )

        semantic_cache = get_semantic_cache()
        if semantic_cache is not None:
            semantic_hit = semantic_cache.lookup(request, active_version)
            if semantic_hit is not None:
                logger.info("estimate.semantic_cache_hit", prompt_version=active_version)
                return StructuredEstimation(
                    result=semantic_hit, model="", provider="", cache_hit=True
                )

        result, meta = get_llm_wrapper().complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=EstimationResult,
            temperature=settings.temperature,
            max_tokens=settings.llm_max_tokens,
            max_retries=settings.structured_max_retries,
        )

        if settings.guardrails_enabled:
            result = enforce_scope_response(result)

        meta_model = str(meta.get("model", ""))
        meta_provider = str(meta.get("provider", ""))
        meta_input: int | None = meta.get("input_tokens")
        meta_output: int | None = meta.get("output_tokens")
        meta_cost: float | None = meta.get("cost_usd")
        meta_fallback = bool(meta.get("fallback_used", False))

        cache.set(
            cache_key,
            {
                "result": result.model_dump(mode="json"),
                "model": meta_model,
                "provider": meta_provider,
                "input_tokens": meta_input,
                "output_tokens": meta_output,
                "cost_usd": meta_cost,
                "fallback_used": meta_fallback,
            },
        )
        if semantic_cache is not None:
            semantic_cache.store(request, result, active_version)

        return StructuredEstimation(
            result=result,
            model=meta_model,
            provider=meta_provider,
            input_tokens=meta_input,
            output_tokens=meta_output,
            cost_usd=meta_cost,
            fallback_used=meta_fallback,
            cache_hit=False,
        )

    # -- Estimación conversacional (sesión 5) --------------------------------

    def estimate_conversational(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        version: str | None = None,
    ) -> SessionEstimation:
        """Turno de estimación conversacional (sesión 5).

        Diferencias con ``estimate``:
        - **Sin caché**: cada turno depende del historial y de la metadata, así que
          dos transcripciones idénticas en sesiones distintas no son la misma llamada.
        - El system prompt se regenera con el bloque `<project_metadata>` actual y el
          modelo recibe además la ventana de historial como array `messages`.
        - Tras validar, el turno se añade al historial y una segunda llamada
          refresca `ProjectMetadata` de forma tolerante a fallos.
        """
        active_version = version or self.conversational_prompt_version
        require_configuration()
        warn_if_temperature_ignored()
        run_input_guardrails(transcript)

        system_prompt, user_message = render_conversational_prompt(
            description=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            metadata=session.metadata,
            version=active_version,
        )
        messages = session.history.to_messages_list(system_prompt)
        messages.append({"role": "user", "content": user_message})

        wrapper = get_llm_wrapper()
        result, meta = wrapper.complete_structured_messages(
            messages=messages,
            response_model=EstimationResult,
            temperature=settings.temperature,
            max_tokens=settings.llm_max_tokens,
            max_retries=settings.structured_max_retries,
        )
        if settings.guardrails_enabled:
            result = enforce_scope_response(result)

        session.history.append(user=user_message, assistant=result.model_dump_json())
        session.metadata = update_metadata(
            previous=session.metadata,
            transcript=transcript,
            result=result,
            llm_wrapper=wrapper,
            model=self.metadata_extractor_model or settings.metadata_extractor_model,
        )

        return SessionEstimation(
            result=result,
            model=str(meta.get("model", "")),
            provider=str(meta.get("provider", "")),
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            cost_usd=meta.get("cost_usd"),
            fallback_used=bool(meta.get("fallback_used", False)),
        )
