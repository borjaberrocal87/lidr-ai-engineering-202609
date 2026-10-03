"""Segunda llamada LLM por turno: extrae `ProjectMetadata` de la interacción.

El pipeline conversacional la ejecuta DESPUÉS de la estimación. La llamada es
estrecha: un modelo barato con un prompt corto que devuelve solo hechos
durables. El resultado se mezcla con la metadata previa y se guarda en la
sesión. Si la extracción falla por cualquier motivo, se registra y se devuelve
la metadata anterior sin cambios: perder un refresco no debe romper el turno.
"""

from __future__ import annotations

import structlog

from app.prompts.loader import render_metadata_extraction_prompt
from app.schemas.estimations import EstimationResult
from app.services.llm_wrapper import LLMWrapper
from app.sessions.models import ProjectMetadata

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def update_metadata(
    *,
    previous: ProjectMetadata,
    transcript: str,
    result: EstimationResult,
    llm_wrapper: LLMWrapper,
    model: str = "",
) -> ProjectMetadata:
    """Extrae metadata del turno y la mezcla con la previa.

    `model` vacío usa el modelo principal del wrapper. Ante cualquier fallo,
    devuelve `previous` sin tocar.
    """
    system_prompt, user_message = render_metadata_extraction_prompt(
        transcript=transcript,
        result=result,
        previous=previous,
    )
    try:
        extracted, meta = llm_wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=ProjectMetadata,
            temperature=0.0,
            max_tokens=1000,
            max_retries=2,
            model_override=model or None,
        )
    except Exception as exc:
        logger.warning(
            "metadata_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return previous

    merged = previous.merge_with(extracted)
    logger.info(
        "metadata_extraction_completed",
        model=str(meta.get("model", "")),
        project_name=merged.project_name,
        team_size=merged.assumed_team_size,
        tech_count=len(merged.mentioned_technologies),
    )
    return merged
