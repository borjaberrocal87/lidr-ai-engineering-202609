"""Orquestador RAG end-to-end (Sesión 9).

Encadena las cuatro etapas que faltaban desde el primer día:
``transcripción → comprensión → retrieval → aumentación → generación``,
produciendo una :class:`GroundedEstimate` con citas. La generación pasa por
``LLMWrapper`` (structured output), nunca por una API cruda.

Las firmas públicas son async; las llamadas síncronas del wrapper y del
embedder se empujan a hilos para no bloquear el event loop.
"""

from __future__ import annotations

import asyncio

import structlog

from app.config import settings
from app.foundation.prompts.loader import (
    render_rag_estimation_prompt,
    render_rag_structure_prompt,
)
from app.generation.rag.brief_extractor import compose_retrieval_text, extract_project_brief
from app.generation.rag.context_builder import assemble_context, fit_chunks_to_budget
from app.generation.rag.errors import (
    IncoherentEstimateError,
    RagGenerationError,
    RagRetrievalError,
)
from app.generation.rag.grounding import find_fabricated_citations, is_coherent_estimate
from app.generation.rag.schemas import GroundedEstimate, ProjectBrief, RetrievalOutcome
from app.generation.rag.stage_logging import current_request_id, log_pipeline_stage

log = structlog.get_logger()

# Sectores presentes en el corpus; solo filtramos el retrieval cuando el brief
# nombra uno de ellos (evita sobre-filtrar por valores libres).
_KNOWN_SECTORS = {
    "finance",
    "ecommerce",
    "healthcare",
    "industrial",
    "logistics",
    "education",
    "media",
    "government",
}


async def generate_grounded_estimate(
    context_block: str,
    brief: ProjectBrief,
    *,
    include_hours: bool = True,
) -> GroundedEstimate:
    """Genera una :class:`GroundedEstimate` a partir del contexto ensamblado.

    Las citas las valida el llamante, no esta función. Con ``include_hours=False``
    el modelo propone la estructura sin cifras (S10); las horas se derivan
    después por búsqueda vectorial por tarea.
    """
    return await _generate(context_block, brief, include_hours=include_hours)


async def _generate(
    context_block: str,
    brief: ProjectBrief,
    *,
    feedback: str | None = None,
    include_hours: bool = True,
) -> GroundedEstimate:
    """Una llamada de generación. ``feedback`` añade una nota correctiva."""
    from app.dependencies import get_llm_wrapper

    wrapper = get_llm_wrapper()

    system_prompt, user_message = render_rag_estimation_prompt(
        brief=brief,
        context_block=context_block,
        include_hours=include_hours,
    )
    if feedback:
        user_message += f"\n\n<correction>\n{feedback}\n</correction>"

    try:
        estimate, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=GroundedEstimate,
            temperature=settings.temperature,
            model_override=settings.rag_generation_model,
            reasoning_effort=settings.rag_generation_reasoning_effort,
            max_tokens=settings.rag_generation_max_tokens,
            # El pipeline ya reintenta una vez ante citas inventadas; no
            # duplicamos la espera con los reintentos de esquema del wrapper.
            max_retries=0,
        )
        return estimate
    except Exception as exc:
        raise RagGenerationError("No se pudo generar la estimación grounded.") from exc


async def generate_structure(brief: ProjectBrief) -> GroundedEstimate:
    """Genera la ESTRUCTURA módulo→tarea como descomposición libre del brief (S10).

    No hay retrieval ni fuentes: se apoya en el criterio del modelo sobre lo que
    el proyecto requiere. Devuelve la misma :class:`GroundedEstimate` (con
    ``engineer_days`` y ``total_engineer_days`` a null y ``sources`` vacías) para
    que el wizard la parsee sin un contrato nuevo. No se valida citas porque no
    hay fuentes que validar.
    """
    from app.dependencies import get_llm_wrapper

    wrapper = get_llm_wrapper()
    system_prompt, user_message = render_rag_structure_prompt(brief)

    try:
        estimate, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=GroundedEstimate,
            temperature=settings.temperature,
            model_override=settings.rag_generation_model,
            reasoning_effort=settings.rag_generation_reasoning_effort,
            max_tokens=settings.rag_generation_max_tokens,
            max_retries=0,
        )
        return estimate
    except Exception as exc:
        raise RagGenerationError("No se pudo generar la estructura.") from exc


def _insufficient(explanation: str) -> GroundedEstimate:
    """Construye la estimación canónica de contexto insuficiente (sin cifras)."""
    return GroundedEstimate(
        total_engineer_days=None,
        duration_weeks=None,
        confidence="insufficient",
        reasoning="El retrieval no encontró presupuestos históricos suficientemente relevantes.",
        insufficient_context_explanation=explanation,
    )


async def _retrieve(transcript_brief: ProjectBrief) -> RetrievalOutcome:
    """Embed the search text and run filtered retrieval with S10 toggles."""
    from app.dependencies import get_runtime_retrieval_config, get_semantic_retriever

    retriever = get_semantic_retriever()
    if retriever is None:
        raise RagRetrievalError("Embedding service is not configured.")

    retrieval_text = compose_retrieval_text(transcript_brief)
    sector = transcript_brief.sector.lower().strip() if transcript_brief.sector else None
    sectors = [sector] if sector in _KNOWN_SECTORS else None

    runtime = get_runtime_retrieval_config()
    search_mode = runtime.effective_search_mode()
    rerank = runtime.effective_rerank()

    try:
        return await retriever.retrieve_chunks(
            query_text=retrieval_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=settings.rag_retrieval_top_k,
            recall_k=settings.retrieval_recall_top_k,
            rerank_top_n=settings.rerank_top_n,
            distance_threshold=settings.rag_retrieval_distance_threshold,
            rrf_k=settings.rrf_k,
            sectors=sectors,
        )
    except Exception as exc:
        raise RagRetrievalError("Historical budget retrieval failed.") from exc


async def estimate_from_transcript(
    transcript: str,
    idempotency_key: str | None = None,
) -> GroundedEstimate:
    """Ejecuta el pipeline completo transcripción → estimación grounded.

    Pasos: idempotencia → brief → retrieval filtrado (soft-fail a estimación
    insuficiente) → truncado por tokens → contexto → generación → validación de
    citas (un reintento correctivo) → coherencia → caché.
    """
    from app.dependencies import get_idempotency_store, get_token_encoder

    request_id = current_request_id()
    store = get_idempotency_store()

    if idempotency_key:
        cached = await asyncio.to_thread(store.get, idempotency_key)
        if cached is not None:
            log.info(
                "idempotency.hit",
                request_id=request_id,
                idempotency_key=idempotency_key,
            )
            return cached

    # 1. Comprensión de la transcripción.
    with log_pipeline_stage("brief", request_id):
        brief = await extract_project_brief(transcript)

    # 2. Retrieval filtrado con soft-fail.
    with log_pipeline_stage("retrieval", request_id):
        retrieval = await _retrieve(brief)

    if retrieval.low_confidence:
        log.info(
            "retrieval.soft_fail",
            request_id=request_id,
            candidates=retrieval.candidates_evaluated,
        )
        estimate = _insufficient(
            "Ningún presupuesto histórico superó el umbral de relevancia para este proyecto."
        )
        if idempotency_key:
            await asyncio.to_thread(store.set, idempotency_key, estimate)
        return estimate

    # 3. Truncado por presupuesto (solo chunks completos) y ensamblado.
    encoder = get_token_encoder()
    kept = fit_chunks_to_budget(
        retrieval.chunks,
        settings.rag_max_context_tokens,
        encoder,
    )
    context_block = assemble_context(kept)

    # 4. Generación grounded.
    with log_pipeline_stage("generation", request_id, sources=len(kept)):
        estimate = await generate_grounded_estimate(context_block, brief)

    # 5. Validación de citas; un reintento correctivo ante ids inventados.
    fabricated = find_fabricated_citations(estimate, kept)
    if fabricated:
        feedback = (
            f"tu respuesta anterior citó source ids inválidos: {fabricated}. "
            "Cita solo ids que aparezcan en el bloque <sources>."
        )
        with log_pipeline_stage("citation_retry", request_id, fabricated=fabricated):
            estimate = await _generate(context_block, brief, feedback=feedback)
        if find_fabricated_citations(estimate, kept):
            log.warning("citations.unrepaired", request_id=request_id)
            estimate = estimate.model_copy(update={"confidence": "low"})

    # 6. Guarda de coherencia: un intento de reparación y, si falla, rechazo.
    if not is_coherent_estimate(estimate):
        feedback = (
            'cuando confidence sea "insufficient", total_engineer_days y '
            "duration_weeks deben ser null, modules debe estar vacío e "
            "insufficient_context_explanation debe estar relleno; en caso "
            "contrario, aporta módulos, tareas y cifras."
        )
        with log_pipeline_stage("coherence_repair", request_id):
            estimate = await _generate(context_block, brief, feedback=feedback)
        if not is_coherent_estimate(estimate):
            raise IncoherentEstimateError(
                "La estimación incumple la regla de coherencia de contexto insuficiente."
            )

    # 7. Persistencia para reintentos idempotentes.
    if idempotency_key:
        await asyncio.to_thread(store.set, idempotency_key, estimate)

    return estimate
