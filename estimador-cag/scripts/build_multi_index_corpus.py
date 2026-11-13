#!/usr/bin/env python3
"""Ingesta las fuentes de transcripciones y documentación técnica (S10).

El corpus de presupuestos se ingiere por HTTP con ``query_examples.py``. Este
script siembra las dos fuentes nuevas del catálogo (``transcript_chunks`` y
``technical_doc_chunks``) directamente a través de ``ChunkStore``, reutilizando
el embedder y la factoría de sesiones del proyecto. Un chunk por segmento de
reunión y por sección de documento, con la metadata propia de cada fuente
(transcripciones: hablantes y fecha; documentos: versión), de modo que el
enrutado, los filtros duros y la vigencia devuelvan resultados con sentido.

Idempotente: cada documento se identifica por ``source_path`` y se salta si ya
está ingerido, así que reejecutar nunca duplica. Para vaciar::

    DELETE FROM documents
     WHERE document_type IN ('meeting_transcript', 'technical_doc');

Ejecutar DENTRO del contenedor (macOS/arm64 en el host choca con el problema de
greenlet al acceder directamente a pgvector)::

    docker compose run --rm estimator python scripts/build_multi_index_corpus.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import get_chunk_store, get_token_encoder  # noqa: E402
from app.foundation.persistence.database import get_async_session_factory  # noqa: E402
from app.generation.rag.retrieval.sources import Collection, spec_for  # noqa: E402
from app.generation.rag.schemas import Chunk  # noqa: E402
from scripts.rag_ops.common import require_embedder  # noqa: E402

TRANSCRIPTS_PATH = ROOT / "data" / "transcripts_sample.json"
DOCS_PATH = ROOT / "data" / "technical_docs_sample.json"


def _chunk(encoder, *, chunk_id: str, header: str, body: str, metadata: dict) -> Chunk:
    """Construye un chunk embebible: una cabecera contextual + el cuerpo."""
    text = f"{header}\n{body}".strip()
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=metadata,
        token_count=len(encoder.encode(text)),
    )


def _transcript_chunks(encoder, transcript: dict) -> list[Chunk]:
    tid = transcript["transcript_id"]
    header = f"Meeting: {transcript['project']} ({transcript['meeting_date']})"
    chunks = []
    for segment in transcript["segments"]:
        chunks.append(
            _chunk(
                encoder,
                chunk_id=f"{tid}::{segment['segment_id']}",
                header=f"{header} — topic: {segment['topic']}",
                body=segment["text"],
                metadata={
                    "transcript_id": tid,
                    "segment_id": segment["segment_id"],
                    "project": transcript["project"],
                    "meeting_date": transcript["meeting_date"],
                    "speakers": segment.get("speakers", []),
                    "topic": segment["topic"],
                },
            )
        )
    return chunks


def _doc_chunks(encoder, doc: dict) -> list[Chunk]:
    did = doc["doc_id"]
    header = f"Doc: {doc['title']} (version {doc['version']})"
    chunks = []
    for section in doc["sections"]:
        chunks.append(
            _chunk(
                encoder,
                chunk_id=f"{did}::{section['section_id']}",
                header=f"{header} — {section['heading']}",
                body=section["text"],
                metadata={
                    "doc_id": did,
                    "section_id": section["section_id"],
                    "heading": section["heading"],
                    "version": doc["version"],
                    "component": doc.get("component", "unknown"),
                },
            )
        )
    return chunks


async def _ingest_documents(
    *,
    label: str,
    items: list[dict],
    source_file: str,
    document_type: str,
    collection: Collection,
    build_chunks,
) -> None:
    """Embebe y persiste cada documento de una fuente, saltando duplicados."""
    embedder = require_embedder()
    encoder = get_token_encoder()
    store = get_chunk_store()
    session_factory = get_async_session_factory()
    spec = spec_for(collection)

    created, skipped = 0, 0
    for item in items:
        doc_key = item.get("transcript_id") or item.get("doc_id")
        source_path = f"{source_file}::{doc_key}"

        async with session_factory() as session:
            if await store.find_document_id(session, source_path) is not None:
                skipped += 1
                continue

        chunks = build_chunks(encoder, item)
        embedded = embedder.embed_many(chunks)

        async with session_factory() as session, session.begin():
            await store.persist_document_with_chunks(
                session,
                source_path=source_path,
                document_type=document_type,
                doc_metadata={"source_id": doc_key},
                embedded_chunks=embedded,
                chunk_type=spec.chunk_type,
                model=spec.model,
            )
        created += 1

    print(f"{label}: {len(items)} documentos — {created} ingeridos, {skipped} ya presentes.")


async def main() -> int:
    transcripts = json.loads(TRANSCRIPTS_PATH.read_text(encoding="utf-8"))
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))

    await _ingest_documents(
        label="Transcripciones",
        items=transcripts,
        source_file="data/transcripts_sample.json",
        document_type="meeting_transcript",
        collection=Collection.TRANSCRIPT,
        build_chunks=_transcript_chunks,
    )
    await _ingest_documents(
        label="Documentación técnica",
        items=docs,
        source_file="data/technical_docs_sample.json",
        document_type="technical_doc",
        collection=Collection.TECHNICAL_DOC,
        build_chunks=_doc_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
