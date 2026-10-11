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
from app.schemas.critic import CriticFeedback
from app.schemas.estimations import (
    ACBResponse,
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
from app.sessions.compression import apply_compression
from app.sessions.metadata_extractor import update_metadata
from app.sessions.models import Session
from app.sessions.tier_resolver import Tier, resolve_tier

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
        conversational_prompt_version: str | None = None,
        metadata_extractor_model: str | None = None,
        critic_model: str | None = None,
        boss_max_iterations: int | None = None,
    ) -> None:
        self.prompt_version = prompt_version
        self.conversational_prompt_version = conversational_prompt_version
        self.metadata_extractor_model = metadata_extractor_model
        self.critic_model = critic_model
        self.boss_max_iterations = boss_max_iterations

    def _conversational_version(self, version: str | None) -> str:
        return (
            version
            or self.conversational_prompt_version
            or settings.conversational_prompt_version
            or DEFAULT_CONVERSATIONAL_PROMPT_VERSION
        )

    def _compression_model(self) -> str:
        return settings.compression_model or settings.llm_model

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
        tier: Tier | None = None,
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
        active_version = self._conversational_version(version)
        require_configuration()
        warn_if_temperature_ignored()
        run_input_guardrails(transcript)

        # Resolver el tier de audiencia (override explícito o cadena de reglas).
        resolved_tier, rule = resolve_tier(
            transcript=transcript,
            metadata=session.metadata,
            override=tier,
        )
        session.last_resolved_tier = resolved_tier.value
        session.last_tier_rule = rule

        system_prompt, user_message = render_conversational_prompt(
            description=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            metadata=session.metadata,
            version=active_version,
            tier=resolved_tier,
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
        apply_compression(
            session.history,
            llm_wrapper=wrapper,
            compression_model=self._compression_model(),
            anchor_detection_mode=settings.anchor_detection_mode,
        )
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

    # -- Estimación Actor-Critic-Boss (sesión 5, directo) --------------------

    def estimate_with_acb(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        tier: Tier | None = None,
        version: str | None = None,
    ) -> ACBResponse:
        """Variante Actor-Critic-Boss de la pipeline conversacional.

        La sesión se actualiza **solo** con el resultado final aprobado (o
        sintetizado) por el Boss — los borradores intermedios del actor se
        descartan. Así el estado de la conversación se mantiene coherente: desde
        el punto de vista del usuario, el turno produjo exactamente un mensaje
        del asistente.
        """
        from app.services.boss import Boss
        from app.services.critic import Critic

        active_version = self._conversational_version(version)

        # 1. Guardrail de entrada.
        require_configuration()
        warn_if_temperature_ignored()
        run_input_guardrails(transcript)

        # 2. Resolver tier (igual que el camino del actor).
        resolved_tier, rule = resolve_tier(
            transcript=transcript,
            metadata=session.metadata,
            override=tier,
        )
        session.last_resolved_tier = resolved_tier.value
        session.last_tier_rule = rule

        logger.info(
            "estimation_acb_request",
            session_id=session.session_id,
            tier=resolved_tier.value,
            tier_rule=rule,
            transcript_chars=len(transcript),
        )

        wrapper = get_llm_wrapper()

        # 3. Construir el callable del actor. Re-renderiza el prompt en cada
        #    iteración para entretejer el feedback del critic (si lo hay). El
        #    guardrail de salida corre en cada borrador; si el Boss acaba
        #    aceptando un borrador, ese mismo resultado (ya guardrailizado) es el
        #    que se persiste en la sesión más abajo.
        def _actor(critic_feedback: CriticFeedback | None) -> EstimationResult:
            system_prompt, user_message = render_conversational_prompt(
                description=transcript,
                project_type=project_type,
                detail_level=detail_level,
                output_format=output_format,
                metadata=session.metadata,
                version=active_version,
                tier=resolved_tier,
                critic_feedback=critic_feedback,
            )
            messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            messages.extend(
                {"role": message.role, "content": message.content}
                for message in session.history.messages
            )
            messages.append({"role": "user", "content": user_message})

            draft, meta = wrapper.complete_structured_messages(
                messages=messages,
                response_model=EstimationResult,
                temperature=settings.temperature,
                max_tokens=settings.llm_max_tokens,
                max_retries=settings.structured_max_retries,
            )
            logger.info(
                "acb_actor_draft",
                with_critic_feedback=critic_feedback is not None,
                issues_in_feedback=(
                    len(critic_feedback.issues) if critic_feedback is not None else 0
                ),
                confidence_pct=draft.confidence_pct,
                total_cost_eur=draft.total_cost_eur,
                **meta,
            )
            return enforce_scope_response(draft) if settings.guardrails_enabled else draft

        # 4. Construir el callable del critic.
        critic = Critic(
            llm_wrapper=wrapper,
            model=self.critic_model or settings.critic_model or settings.llm_model,
        )

        def _critic(draft: EstimationResult) -> CriticFeedback:
            return critic.review(
                transcript=transcript,
                metadata=session.metadata,
                tier=resolved_tier,
                result=draft,
            )

        # 5. El Boss orquesta.
        boss = Boss(
            max_iterations=self.boss_max_iterations or settings.boss_max_iterations,
        )
        final_result, trace = boss.run(actor=_actor, critic=_critic)

        # 6. Persistir el resultado final en la sesión (un único turno).
        turn_user_message = render_conversational_prompt(
            description=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            metadata=session.metadata,
            version=active_version,
            tier=resolved_tier,
        )[1]
        session.history.append(user=turn_user_message, assistant=final_result.model_dump_json())
        apply_compression(
            session.history,
            llm_wrapper=wrapper,
            compression_model=self._compression_model(),
            anchor_detection_mode=settings.anchor_detection_mode,
        )

        # 7. Refrescar metadata desde el resultado final.
        session.metadata = update_metadata(
            previous=session.metadata,
            transcript=transcript,
            result=final_result,
            llm_wrapper=wrapper,
            model=self.metadata_extractor_model or settings.metadata_extractor_model,
        )

        return ACBResponse(
            session_id=session.session_id,
            result=final_result,
            prompt_version=active_version,
            acb=trace,
            metadata=session.metadata,
            history_messages=len(session.history.messages),
            model=settings.llm_model,
            provider=settings.llm_provider,
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            fallback_used=False,
        )
