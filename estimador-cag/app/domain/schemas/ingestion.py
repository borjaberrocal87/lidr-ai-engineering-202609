"""Esquemas de Pydantic v2 para la capa HTTP de ingesta.

La API tiene dos endpoints (POST + GET) y tres formas de respuesta. Mantenerlos
en un único módulo hace que el contrato sea fácil de leer junto al código del
router.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestionRunRequest(BaseModel):
    """Cuerpo de ``POST /api/v1/ingestion/runs``."""

    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "Name of an ``include`` source in the catalog. Anything else "
            "(unknown, review, exclude) is rejected with HTTP 400."
        ),
    )


class IngestionRunResponse(BaseModel):
    """Respuesta de ``POST /api/v1/ingestion/runs``. Se devuelve con HTTP 202."""

    job_id: uuid.UUID
    source_name: str
    status: Literal["pending", "running", "completed", "failed"]


class IngestionJobView(BaseModel):
    """Respuesta de ``GET /api/v1/ingestion/jobs/{job_id}``."""

    job_id: uuid.UUID
    source_name: str
    status: Literal["pending", "running", "completed", "failed"]
    documents_count: int
    error_message: str | None
    started_at: datetime
    finished_at: datetime | None


class QualityScoreView(BaseModel):
    """Puntuación por dimensión (1..5) del catálogo."""

    completeness: int
    consistency: int
    actuality: int
    reliability: int


class IngestionSourceView(BaseModel):
    """Vista de una fuente del catálogo para ``GET /api/v1/ingestion/sources``."""

    name: str
    description: str
    format: str
    decision: Literal["include", "review", "exclude"]
    decision_reason: str
    quality: QualityScoreView
    has_pii: bool
    access_level: str


class IngestionSourcesResponse(BaseModel):
    """Respuesta de ``GET /api/v1/ingestion/sources``."""

    version: str
    sources: list[IngestionSourceView]


class IngestionJobsPage(BaseModel):
    """Respuesta de ``GET /api/v1/ingestion/jobs`` (paginado)."""

    items: list[IngestionJobView]
    limit: int
    offset: int
