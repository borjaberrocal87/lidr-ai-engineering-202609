"""Endpoints conversacionales (sesión 5).

- ``POST /api/v1/sessions``               — crea una sesión y devuelve su UUID.
- ``GET  /api/v1/sessions/{session_id}``  — vista de debug (metadata + historial).

El endpoint multi-turno de estimación vive aquí también y se añade más abajo.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_session_store
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionNotFoundError, SessionStore

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID de la nueva sesión conversacional.")


class SessionInfoResponse(BaseModel):
    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata


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
    )
