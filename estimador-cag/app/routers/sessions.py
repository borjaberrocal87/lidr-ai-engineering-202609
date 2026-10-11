"""Endpoints conversacionales (sesión 5).

- ``POST /api/v1/sessions``                       — crea una sesión (UUID).
- ``GET  /api/v1/sessions/{session_id}``          — vista de debug (metadata + historial).
- ``POST /api/v1/sessions/{session_id}/estimate`` — turno multi-turno con adjuntos
  (multipart/form-data). El texto de los adjuntos se extrae localmente (Camino B)
  y se concatena a la transcripción antes de llamar al modelo.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.attachments.extractor import (
    AttachmentExtractionError,
    UnsupportedAttachmentError,
    enrich_transcript,
    extract_text,
)
from app.config import settings
from app.dependencies import get_estimation_service, get_session_store
from app.guardrails.input import InputGuardrailViolation
from app.routers.estimations import validated_conversational_prompt_version
from app.schemas.estimations import (
    ACBResponse,
    DetailLevel,
    OutputFormat,
    ProjectType,
    SessionEstimateResponse,
)
from app.services.errors import LLMConfigurationError, LLMProviderError
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata, Session
from app.sessions.store import SessionNotFoundError, SessionStore
from app.sessions.tier_resolver import Tier

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID de la nueva sesión conversacional.")


class SessionInfoResponse(BaseModel):
    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata
    anchors_count: int = 0
    summary_chars: int = 0
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None


@router.post("", response_model=CreateSessionResponse, status_code=201)
def create_session(
    store: Annotated[SessionStore, Depends(get_session_store)],
) -> CreateSessionResponse:
    session = store.create()
    logger.info("session.created", session_id=session.session_id)
    return CreateSessionResponse(session_id=session.session_id)


@router.get("/{session_id}", response_model=SessionInfoResponse)
def get_session(
    session_id: str,
    store: Annotated[SessionStore, Depends(get_session_store)],
) -> SessionInfoResponse:
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history.messages),
        max_turns=session.history.max_turns,
        metadata=session.metadata,
        anchors_count=len(session.history.anchors),
        summary_chars=len(session.history.summary or ""),
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
    )


@router.post("/{session_id}/estimate", response_model=SessionEstimateResponse)
async def estimate_in_session(
    session_id: str,
    transcript: Annotated[
        str,
        Form(
            min_length=settings.description_min_length,
            max_length=settings.description_max_length,
        ),
    ],
    project_type: Annotated[ProjectType, Form()],
    detail_level: Annotated[DetailLevel, Form()],
    output_format: Annotated[OutputFormat, Form()],
    version: Annotated[str, Depends(validated_conversational_prompt_version)],
    store: Annotated[SessionStore, Depends(get_session_store)],
    service: Annotated[EstimationService, Depends(get_estimation_service)],
    tier: Annotated[Tier | None, Form()] = None,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> SessionEstimateResponse:
    session, enriched = await _load_session_and_enrich(session_id, transcript, attachments, store)
    try:
        outcome = service.estimate_conversational(
            session=session,
            transcript=enriched,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=tier,
            version=version,
        )
    except Exception as exc:
        raise _map_pipeline_errors(exc) from exc

    return SessionEstimateResponse(
        session_id=session.session_id,
        result=outcome.result,
        prompt_version=version,
        metadata=session.metadata,
        history_messages=len(session.history.messages),
        model=outcome.model,
        provider=outcome.provider,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        cost_usd=outcome.cost_usd,
        fallback_used=outcome.fallback_used,
    )


@router.post("/{session_id}/estimate-acb", response_model=ACBResponse)
async def estimate_in_session_acb(
    session_id: str,
    transcript: Annotated[
        str,
        Form(
            min_length=settings.description_min_length,
            max_length=settings.description_max_length,
        ),
    ],
    project_type: Annotated[ProjectType, Form()],
    detail_level: Annotated[DetailLevel, Form()],
    output_format: Annotated[OutputFormat, Form()],
    version: Annotated[str, Depends(validated_conversational_prompt_version)],
    store: Annotated[SessionStore, Depends(get_session_store)],
    service: Annotated[EstimationService, Depends(get_estimation_service)],
    tier: Annotated[Tier | None, Form()] = None,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> ACBResponse:
    """Variante Actor-Critic-Boss de `/estimate`.

    Mismo contrato multipart; la respuesta incluye el campo ``acb`` con la traza
    de iteraciones (veredicto, confianza e issues por ronda) para que el cliente
    pueda mostrar la auditoría en su UI.
    """
    session, enriched = await _load_session_and_enrich(session_id, transcript, attachments, store)
    try:
        return service.estimate_with_acb(
            session=session,
            transcript=enriched,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=tier,
            version=version,
        )
    except Exception as exc:
        raise _map_pipeline_errors(exc) from exc


async def _load_session_and_enrich(
    session_id: str,
    transcript: str,
    attachments: list[UploadFile] | None,
    store: SessionStore,
) -> tuple[Session, str]:
    """Preludio compartido por `/estimate` y `/estimate-acb`.

    Devuelve ``(session, transcript_enriquecido)``. Lanza ``HTTPException`` para
    problemas de sesión o adjuntos; el error del LLM lo mapea el llamante.
    """
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc

    extracted: list[tuple[str, str]] = []
    for upload in attachments or []:
        if not upload.filename:
            continue
        content = await upload.read()
        try:
            text = extract_text(
                filename=upload.filename,
                content=content,
                max_chars=settings.max_attachment_chars,
            )
        except UnsupportedAttachmentError as exc:
            raise HTTPException(
                status_code=415,
                detail={"reason": "unsupported_attachment", "filename": exc.filename},
            ) from exc
        except AttachmentExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "attachment_extraction_failed",
                    "filename": exc.filename,
                    "message": exc.message,
                },
            ) from exc
        if text:
            extracted.append((upload.filename, text))

    enriched = enrich_transcript(transcript=transcript, attachments=extracted)
    logger.info(
        "session.estimate_received",
        session_id=session_id,
        transcript_chars=len(transcript),
        enriched_chars=len(enriched),
        attachment_count=len(extracted),
    )
    return session, enriched


def _map_pipeline_errors(exc: Exception) -> HTTPException:
    """Mapeo común: guardrail de entrada → 400, config → 503, proveedor → 502."""
    if isinstance(exc, InputGuardrailViolation):
        logger.info("session.estimate_guardrail_blocked", reason=exc.reason)
        return HTTPException(status_code=400, detail={"reason": exc.reason, "message": exc.message})
    if isinstance(exc, LLMConfigurationError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, LLMProviderError):
        logger.warning("session.estimate_provider_error", provider=settings.llm_provider)
        return HTTPException(
            status_code=502, detail="No se pudo generar la estimación. Inténtalo de nuevo."
        )
    logger.exception("session.estimate_unexpected_error")
    return HTTPException(status_code=502, detail="Error inesperado generando la estimación.")
