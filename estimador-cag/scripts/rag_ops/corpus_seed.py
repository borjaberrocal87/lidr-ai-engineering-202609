#!/usr/bin/env python3
"""Genera corpus sintético con embeddings reales para probar a escala.

Los vectores aleatorios no sirven para medir recall: no comparten la geometría
del corpus. Este script genera textos variados a partir de plantillas, los
embebe con el mismo modelo del proyecto y los persiste bajo un documento
marcado como ``synthetic_test``.

Casos de uso:

* ``100`` chunks: demo del ciclo de mantenimiento (ANALYZE/VACUUM/REINDEX).
* ``30000`` chunks: preflight opcional para que el baseline sin índice sea
  medible. Tiene coste real de embeddings y tarda varios minutos.

Uso::

    docker compose exec api python -m scripts.rag_ops.corpus_seed
    docker compose exec api python -m scripts.rag_ops.corpus_seed 30000 --dry-run
    docker compose exec api python -m scripts.rag_ops.corpus_seed --reset

Limpieza: ``DELETE FROM documents WHERE source_path = 'synthetic::rag-ops';``
(la FK con ``ON DELETE CASCADE`` borra también sus chunks).
"""

from __future__ import annotations

import argparse
import asyncio
import random

import tiktoken
from sqlalchemy import select, text

from app.embedding_pipeline.embedder import estimate_cost_usd
from app.embedding_pipeline.schemas import Chunk
from app.embedding_pipeline.store.models import ChunkRow, DocumentRow
from app.persistence.database import get_async_session_factory
from scripts.rag_ops.common import (
    print_footer,
    print_header,
    require_embedder,
    run_async,
)

DEFAULT_SOURCE_PATH = "synthetic::rag-ops"
DEFAULT_DOCUMENT_TYPE = "synthetic_test"
DEFAULT_CHUNK_TYPE = "synthetic"
DEFAULT_BATCH_SIZE = 500

TEMPLATES = (
    "Componente de {feature} para el sector {sector}. Incluye {detail} y {constraint}.",
    "Módulo de {feature} para una plataforma del sector {sector}. "
    "Requiere {detail}; se asume {constraint}.",
    "Servicio de {feature} con foco en {sector}. Entregable: {detail}. Restricción: {constraint}.",
    "Trabajo de {feature} para cliente de {sector}. Alcance: {detail}. "
    "Calidad exigida: {constraint}.",
)

FEATURES = (
    "autenticación OAuth",
    "API REST",
    "procesamiento batch",
    "streaming de eventos",
    "panel de analítica",
    "migración de datos",
    "motor de reglas",
    "integración de pagos",
    "búsqueda semántica",
    "orquestación de workflows",
)

SECTORS = (
    "fintech",
    "salud",
    "retail",
    "logística",
    "seguros",
    "educación",
    "energía",
    "administración pública",
)

DETAILS = (
    "diseño de esquema y validación",
    "pruebas de integración con proveedores",
    "observabilidad y trazas distribuidas",
    "estrategia de reintentos e idempotencia",
    "despliegue con contenedores",
    "documentación técnica y runbook",
    "control de acceso basado en roles",
    "optimización de consultas y caché",
)

CONSTRAINTS = (
    "latencia p95 menor de 300 ms",
    "cobertura de tests superior al 80%",
    "cumplimiento GDPR para datos personales",
    "alta disponibilidad 99,9%",
    "auditoría de accesos y trazas",
    "compatibilidad con el sistema legado",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inserta chunks sintéticos con embeddings reales.",
    )
    parser.add_argument("count", nargs="?", type=int, default=100, help="Número de chunks.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Chunks por transacción.",
    )
    parser.add_argument("--seed", type=int, default=8, help="Semilla del generador.")
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--document-type", default=DEFAULT_DOCUMENT_TYPE)
    parser.add_argument("--chunk-type", default=DEFAULT_CHUNK_TYPE)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra el documento sintético previo antes de sembrar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula tokens y coste sin llamar a la API ni tocar la base de datos.",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count debe ser >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size debe ser >= 1")
    return args


def _build_chunks(count: int, *, seed: int) -> list[Chunk]:
    encoding = tiktoken.get_encoding("cl100k_base")
    rng = random.Random(seed)  # noqa: S311 — datos sintéticos, no criptografía
    chunks: list[Chunk] = []

    for index in range(count):
        template = rng.choice(TEMPLATES)
        text_value = template.format(
            feature=rng.choice(FEATURES),
            sector=rng.choice(SECTORS),
            detail=rng.choice(DETAILS),
            constraint=rng.choice(CONSTRAINTS),
        )
        text_value = f"{text_value} Referencia sintética {index:06d}."
        chunks.append(
            Chunk(
                chunk_id=f"synthetic::{index:06d}",
                text=text_value,
                metadata={"synthetic": "true", "index": index},
                token_count=len(encoding.encode(text_value)),
            )
        )
    return chunks


async def _prepare_document(
    source_path: str,
    document_type: str,
    *,
    reset: bool,
) -> int | None:
    session_factory = get_async_session_factory()
    async with session_factory() as session, session.begin():
        existing = (
            await session.execute(
                select(DocumentRow.id).where(DocumentRow.source_path == source_path).limit(1)
            )
        ).scalar_one_or_none()

        if existing is not None:
            if not reset:
                print(
                    f"Ya existe el documento {source_path!r} (id={existing}). "
                    "Usa --reset para recrearlo."
                )
                return None
            await session.execute(
                text("DELETE FROM documents WHERE source_path = :source_path"),
                {"source_path": source_path},
            )

        document = DocumentRow(
            source_path=source_path,
            document_type=document_type,
            metadata_={"origin": "rag_ops", "synthetic": "true"},
        )
        session.add(document)
        await session.flush()
        return document.id


async def _insert_batches(
    document_id: int,
    chunks: list[Chunk],
    *,
    batch_size: int,
    chunk_type: str,
) -> int:
    embedder = require_embedder()
    session_factory = get_async_session_factory()
    inserted = 0
    batches = (len(chunks) + batch_size - 1) // batch_size

    for number, offset in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[offset : offset + batch_size]
        embedded = await asyncio.to_thread(embedder.embed_many, batch)
        async with session_factory() as session, session.begin():
            session.add_all(
                ChunkRow(
                    document_id=document_id,
                    chunk_type=chunk_type,
                    content=chunk.text,
                    embedding=chunk.embedding,
                    metadata_={**chunk.metadata, "batch": number},
                )
                for chunk in embedded
            )
        inserted += len(embedded)
        print(f"  lote {number}/{batches}: {inserted}/{len(chunks)} chunks")

    return inserted


async def _main(args: argparse.Namespace) -> None:
    print_header("Corpus sintético para RAG Ops")
    chunks = _build_chunks(args.count, seed=args.seed)
    total_tokens = sum(chunk.token_count for chunk in chunks)
    estimated_cost = estimate_cost_usd(total_tokens)

    print(
        f"Chunks: {len(chunks)} · tokens: {total_tokens:,} · coste estimado: ${estimated_cost:.4f}"
    )
    print(f"Documento: {args.source_path!r} ({args.document_type})")
    if args.dry_run:
        print("\nDry run: no se llama a la API ni se escribe en Postgres.")
        print(f"Primer chunk: {chunks[0].text}")
        print(f"Último chunk: {chunks[-1].text}")
        return

    document_id = await _prepare_document(
        args.source_path,
        args.document_type,
        reset=args.reset,
    )
    if document_id is None:
        return

    inserted = await _insert_batches(
        document_id,
        chunks,
        batch_size=args.batch_size,
        chunk_type=args.chunk_type,
    )
    print_footer(
        f"Insertados {inserted} chunks en document_id={document_id}.\n"
        f"Limpieza: borra el documento con source_path={args.source_path!r} "
        "(la FK elimina sus chunks en cascada)."
    )


if __name__ == "__main__":
    run_async(_main(_parse_args()))
