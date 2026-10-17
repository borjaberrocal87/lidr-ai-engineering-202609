"""Endpoints de ingesta:

- ``GET  /api/v1/ingestion/sources``          — listado del catálogo (con filtro).
- ``POST /api/v1/ingestion/runs``             — lanza una ingesta (202 + job_id).
- ``GET  /api/v1/ingestion/jobs``             — listado paginado de jobs.
- ``GET  /api/v1/ingestion/jobs/{job_id}``    — estado de un job.

El router es deliberadamente fino: valida la fuente contra el catálogo a través
del caso de uso (``ensure_ingestible``), registra la fila del job, despacha el
orquestador como ``BackgroundTask`` y responde 202 de inmediato. Toda la lógica
de ingesta vive en ``app.ingestion.orchestrator`` — el router nunca importa
parsers y traduce los errores de dominio a HTTP en un único punto.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.config import settings
from app.dependencies import (
    get_catalog,
    get_filesystem_loader,
    get_parser_registry,
    get_text_pseudonymizer,
    get_unit_of_work,
)
from app.ingestion.catalog.models import CatalogSource, DataCatalog
from app.ingestion.errors import SourceNotIncluded, UnknownSource
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.orchestrator import ensure_ingestible, ingest_source
from app.ingestion.parsers.registry import ParserRegistry
from app.persistence.repositories.jobs import Job
from app.persistence.unit_of_work import UnitOfWork
from app.schemas.ingestion import (
    IngestionJobsPage,
    IngestionJobView,
    IngestionRunRequest,
    IngestionRunResponse,
    IngestionSourcesResponse,
    IngestionSourceView,
    QualityScoreView,
)

log = structlog.get_logger()

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def _job_view(job: Job) -> IngestionJobView:
    return IngestionJobView(
        job_id=job.job_id,
        source_name=job.source_name,
        status=job.status,
        documents_count=job.documents_count,
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _source_view(source: CatalogSource) -> IngestionSourceView:
    return IngestionSourceView(
        name=source.name,
        description=source.description,
        format=source.format,
        decision=source.decision.value,
        decision_reason=source.decision_reason,
        quality=QualityScoreView(
            completeness=source.quality.completeness,
            consistency=source.quality.consistency,
            actuality=source.quality.actuality,
            reliability=source.quality.reliability,
        ),
        has_pii=source.sensitivity.has_pii,
        access_level=source.sensitivity.access_level,
    )


def _run_in_background(
    *,
    job_id: uuid.UUID,
    source: CatalogSource,
    source_version: str,
    loader: FileSystemLoader,
    registry: ParserRegistry,
    pseudonymize: bool,
) -> None:
    """Cuerpo de la BackgroundTask. Abre su propio UoW (la sesión de la request
    ya está cerrada cuando esto corre)."""
    uow = UnitOfWork()
    try:
        # El pseudonimizador se construye aquí porque su mapping store va ligado
        # a la sesión del UoW del background, no a la de la request.
        pseudonymizer = get_text_pseudonymizer(uow.session) if pseudonymize else None
        ingest_source(
            source=source,
            source_version=source_version,
            loader=loader,
            registry=registry,
            uow=uow,
            job_id=job_id,
            pseudonymizer=pseudonymizer,
        )
    except Exception as exc:
        # El orquestador ya escribió la fila de fallo; aquí sólo logueamos fuerte.
        log.error(
            "ingestion_background_failed",
            job_id=str(job_id),
            source_name=source.name,
            error=str(exc)[:400],
        )
    finally:
        uow.close()


@router.post("/runs", response_model=IngestionRunResponse, status_code=202)
def create_ingestion_run(
    request: IngestionRunRequest,
    background: BackgroundTasks,
    catalog: Annotated[DataCatalog, Depends(get_catalog)],
    loader: Annotated[FileSystemLoader, Depends(get_filesystem_loader)],
    registry: Annotated[ParserRegistry, Depends(get_parser_registry)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
) -> IngestionRunResponse:
    try:
        source = ensure_ingestible(catalog, request.source_name)
    except UnknownSource as exc:
        raise HTTPException(
            status_code=404,
            detail={"reason": "unknown_source", "source_name": exc.source_name},
        ) from exc
    except SourceNotIncluded as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "source_not_included",
                "source_name": exc.source_name,
                "decision": exc.decision,
                "decision_reason": exc.decision_reason,
            },
        ) from exc

    job = uow.jobs.create(source_name=request.source_name)
    uow.commit()

    background.add_task(
        _run_in_background,
        job_id=job.job_id,
        source=source,
        source_version=catalog.version,
        loader=loader,
        registry=registry,
        pseudonymize=settings.ingestion_pseudonymize_pii and source.sensitivity.has_pii,
    )
    return IngestionRunResponse(job_id=job.job_id, source_name=job.source_name, status=job.status)


@router.get("/sources", response_model=IngestionSourcesResponse)
def list_ingestion_sources(
    catalog: Annotated[DataCatalog, Depends(get_catalog)],
    decision: Annotated[str | None, Query(pattern="^(include|review|exclude)$")] = None,
) -> IngestionSourcesResponse:
    """Lista las fuentes del catálogo, con filtro opcional por decisión."""
    sources = catalog.sources
    if decision is not None:
        sources = [s for s in sources if s.decision.value == decision]
    return IngestionSourcesResponse(
        version=catalog.version,
        sources=[_source_view(s) for s in sources],
    )


@router.get("/jobs", response_model=IngestionJobsPage)
def list_ingestion_jobs(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> IngestionJobsPage:
    """Lista paginada de jobs, del más reciente al más antiguo."""
    jobs = uow.jobs.list(limit=limit, offset=offset)
    return IngestionJobsPage(items=[_job_view(job) for job in jobs], limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=IngestionJobView)
def get_ingestion_job(
    job_id: uuid.UUID,
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
) -> IngestionJobView:
    job = uow.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_view(job)
