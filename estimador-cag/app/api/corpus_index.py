"""Capa HTTP de la expansión incremental del corpus (Sesión 11, directo).

Tres endpoints bajo ``/embeddings/index``:

* ``POST /runs``      — añade un lote de documentos nuevos. Registra el job y
  despacha la expansión como ``BackgroundTask``; responde 202 con el ``job_id``.
* ``GET  /jobs/{id}`` — sondea el avance del job (documentos procesados/estado).
* ``GET  /stats``     — tamaño por colección y si su índice HNSW está creado, para
  que la UI muestre el corpus creciendo tras la expansión.

Los documentos nuevos se embeben con el MISMO modelo del corpus y caen en su
tabla, donde el índice HNSW de la migración 0005 los recoge de forma incremental.
El job vive en Postgres; la expansión corre en el event loop async y sus
mutaciones de job usan una sesión síncrona corta propia (nunca la de la request).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_chunk_store, get_corpus_index_service
from app.foundation.persistence.database import (
    SessionLocal,
    get_async_session_factory,
    get_session,
)
from app.foundation.persistence.repositories.jobs import JobsRepository
from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.schemas import (
    Budget,
    CollectionStats,
    CorpusStats,
    IndexJobView,
    IndexRunRequest,
    IndexRunResponse,
)
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["corpus-index"])


def _record_job(
    job_id: uuid.UUID,
    event: str,
    *,
    documents_count: int = 0,
    error_message: str | None = None,
) -> None:
    """Mutación del job en su propia sesión corta (fuera de la request).

    Abre ``SessionLocal`` por mutación: la BackgroundTask corre después de que la
    sesión de la request se haya cerrado, y la expansión usa el engine async, así
    que ambos extremos no comparten sesión ni loop.
    """
    session = SessionLocal()
    try:
        repo = JobsRepository(session)
        if event == "running":
            repo.mark_running(job_id)
        elif event == "progress":
            repo.set_documents_count(job_id, documents_count)
        elif event == "completed":
            repo.mark_completed(job_id, documents_count=documents_count)
        else:
            repo.mark_failed(job_id, error_message=error_message or "unknown error")
        session.commit()
    finally:
        session.close()


async def _run_expansion(
    *,
    job_id: uuid.UUID,
    documents: list[Budget],
    document_type: str,
    chunk_type: str,
    service: CorpusIndexService,
) -> None:
    """Cuerpo de la BackgroundTask: expande el corpus reportando avance."""

    def report(processed: int) -> None:
        # Escrituras breves a escala de formación; mantienen el contador vivo.
        _record_job(job_id, "progress", documents_count=processed)

    await asyncio.to_thread(_record_job, job_id, "running")
    try:
        result = await service.expand(
            documents,
            document_type=document_type,
            chunk_type=chunk_type,
            on_progress=report,
        )
        await asyncio.to_thread(
            _record_job, job_id, "completed", documents_count=result.documents_indexed
        )
        if result.documents_failed:
            log.warning(
                "corpus_expansion_partial",
                job_id=str(job_id),
                failed=result.documents_failed,
            )
    except Exception as exc:
        message = str(exc)
        log.error("corpus_expansion_failed", job_id=str(job_id), error=message[:400])
        await asyncio.to_thread(_record_job, job_id, "failed", error_message=message)


@router.post("/index/runs", response_model=IndexRunResponse, status_code=202)
def create_index_run(
    request: IndexRunRequest,
    background: BackgroundTasks,
    session: Annotated[Session, Depends(get_session)],
    service: Annotated[CorpusIndexService | None, Depends(get_corpus_index_service)],
) -> IndexRunResponse:
    """Acepta un lote de documentos nuevos y devuelve 202 + job_id."""
    if service is None:
        log.error("corpus_index_unavailable", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    job = JobsRepository(session).create(source_name=f"corpus-expansion:{request.chunk_type}")
    session.commit()

    background.add_task(
        _run_expansion,
        job_id=job.job_id,
        documents=request.documents,
        document_type=request.document_type,
        chunk_type=request.chunk_type,
        service=service,
    )
    log.info(
        "corpus_index_run_created",
        job_id=str(job.job_id),
        documents_total=len(request.documents),
        chunk_type=request.chunk_type,
    )
    return IndexRunResponse(
        job_id=job.job_id,
        documents_total=len(request.documents),
        status=job.status,
    )


@router.get("/index/jobs/{job_id}", response_model=IndexJobView)
def get_index_job(
    job_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> IndexJobView:
    """Estado y progreso de un job de expansión."""
    job = JobsRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return IndexJobView(
        job_id=job.job_id,
        status=job.status,
        documents_processed=job.documents_count,
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/index/stats", response_model=CorpusStats)
async def get_corpus_stats(
    store: Annotated[ChunkStore, Depends(get_chunk_store)],
) -> CorpusStats:
    """Tamaño por colección + estado del índice HNSW (panel de crecimiento)."""
    factory = get_async_session_factory()
    async with factory() as session:
        rows = await store.corpus_stats(session)
    collections = [
        CollectionStats(
            collection=row.collection,
            documents=row.documents,
            chunks=row.chunks,
            hnsw_indexed=row.hnsw_indexed,
        )
        for row in rows
    ]
    return CorpusStats(
        collections=collections,
        total_chunks=sum(collection.chunks for collection in collections),
    )
