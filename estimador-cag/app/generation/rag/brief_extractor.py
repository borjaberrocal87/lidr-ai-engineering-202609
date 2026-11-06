"""Comprensión de la transcripción: de reunión cruda a brief estructurado.

Una transcripción NO es una query. Embeber cientos de tokens de digresiones
produce un vector que es la media semántica de varias intenciones y ruido
conversacional: distancias comprimidas y hits irrelevantes. Esta etapa destila
la transcripción en un :class:`ProjectBrief` y compone después una frase corta,
técnica y alineada con el corpus.

La extracción primaria y el fallback degradado pasan por ``LLMWrapper``, que ya
gestiona reintentos, fallback, coste y re-prompting de structured output.
"""

from __future__ import annotations

import asyncio

import structlog

from app.config import settings
from app.generation.rag.errors import BriefExtractionError
from app.generation.rag.schemas import ProjectBrief

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a software-delivery analyst. Extract a structured project brief from "
    "a raw, messy client meeting transcript. Capture ONLY what the client wants to "
    "build and the constraints that bound it; ignore small talk, anecdotes and "
    "digressions. Normalise everything to concise technical English regardless of "
    "the transcript language. Leave a field empty/unknown when the transcript gives "
    "no evidence for it — never invent technologies, sectors or regulations. "
    "Return the result as a valid JSON object matching the requested schema."
)

_FALLBACK_SYSTEM_PROMPT = (
    "Rewrite the following client meeting transcript as a single short technical "
    "search query in English describing the software to build. One sentence, no "
    "preamble."
)


async def extract_project_brief(transcript: str) -> ProjectBrief:
    """Destila una transcripción en un :class:`ProjectBrief`.

    En la ruta primaria se rellenan todos los campos que el modelo puede
    fundamentar; en el fallback degradado solo se fija ``functionality``.
    """
    from app.dependencies import get_llm_wrapper

    wrapper = get_llm_wrapper()

    try:
        brief, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=_SYSTEM_PROMPT,
            user_message=transcript,
            response_model=ProjectBrief,
            temperature=settings.temperature,
            model_override=settings.rag_brief_model,
            max_tokens=settings.rag_brief_max_tokens,
        )
        return brief
    except Exception as exc:
        log.warning(
            "brief.fallback",
            reason="structured_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )

    try:
        result = await asyncio.to_thread(
            wrapper.complete,
            system_prompt=_FALLBACK_SYSTEM_PROMPT,
            user_message=transcript,
            temperature=settings.temperature,
            max_tokens=settings.rag_brief_max_tokens,
        )
        rewritten = (result.get("estimation") or "").strip()
        if not rewritten:
            raise BriefExtractionError("El fallback de reescritura devolvió un texto vacío.")
        return ProjectBrief(functionality=rewritten)
    except BriefExtractionError:
        raise
    except Exception as exc:
        raise BriefExtractionError("No se pudo extraer el brief de la transcripción.") from exc


def compose_retrieval_text(brief: ProjectBrief) -> str:
    """Compone la frase técnica que se embebe a partir del brief.

    Une los campos fundamentados en una frase alineada con el corpus, p. ej.
    ``"B2B payments marketplace platform with Stripe Connect, KYC, SAP for
    healthcare in Germany, BaFin-compliant"``. Los campos vacíos se omiten.
    """
    parts: list[str] = [brief.functionality.strip()] if brief.functionality.strip() else []

    if brief.technologies:
        parts.append("with " + ", ".join(brief.technologies))
    if brief.sector:
        parts.append(f"for {brief.sector}")
    if brief.country:
        parts.append(f"in {brief.country}")
    if brief.regulations:
        regulations = ", ".join(brief.regulations) + "-compliant"
        if parts:
            parts[-1] = f"{parts[-1]}, {regulations}"
        else:
            parts.append(regulations)

    return " ".join(parts).strip()
