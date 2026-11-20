#!/usr/bin/env python3
"""Verifica que los índices HNSW de la Sesión 11 existen, están alineados y se usan.

Para cada colección del store vectorial (``budget_chunks``, ``transcript_chunks``
y ``technical_doc_chunks``) comprueba:

1. que exista el índice que crea la migración ``0005_session11_hnsw_multi_index``;
2. que esté construido sobre ``(embedding::halfvec(1536))`` con la operator class
   ``halfvec_cosine_ops`` — la que usa el retriever con ``SEARCH_USE_HALFVEC=true``;
3. que el planner pueda elegirlo: ejecuta el ``ORDER BY`` halfvec bajo
   ``SET enable_seqscan=off`` y ``SET jit=off`` (solo en esta sesión de psql).

Sobre el seq scan: en un corpus pequeño (1-2k chunks) el planner prefiere
recorrer la tabla antes que arrancar el grafo HNSW, y eso es correcto. Este
script fuerza ``enable_seqscan=off`` únicamente para DEMOSTRAR que el índice está
bien formado y es utilizable; no cambia configuración persistente ni la app.
Cuando el corpus crezca, el planificador lo elegirá solo.

Uso::

    docker compose exec api python -m scripts.rag_ops.explain_index
    docker compose exec api python -m scripts.rag_ops.explain_index --no-explain
    docker compose exec api python -m scripts.rag_ops.explain_index --json

Sale con código 1 si falta un índice, no está alineado o el plan no lo usa.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

TABLES = ("budget_chunks", "transcript_chunks", "technical_doc_chunks")
EMBEDDING_DIMENSIONS = 1536
EXPECTED_OPERATOR_CLASS = "halfvec_cosine_ops"

# SQL fijo (sin interpolación de entrada): una entrada por tabla.
_INDEX_SQL = "SELECT indexdef FROM pg_indexes WHERE indexname = :index_name LIMIT 1"
_COUNT_SQL: dict[str, str] = {
    "budget_chunks": "SELECT count(*) FROM budget_chunks",
    "transcript_chunks": "SELECT count(*) FROM transcript_chunks",
    "technical_doc_chunks": "SELECT count(*) FROM technical_doc_chunks",
}
_EXPLAIN_SQL: dict[str, str] = {
    "budget_chunks": (
        "EXPLAIN (ANALYZE, FORMAT JSON) SELECT id FROM budget_chunks "
        "ORDER BY embedding::halfvec(1536) <=> "
        "(SELECT embedding::halfvec(1536) FROM budget_chunks LIMIT 1) LIMIT 5"
    ),
    "transcript_chunks": (
        "EXPLAIN (ANALYZE, FORMAT JSON) SELECT id FROM transcript_chunks "
        "ORDER BY embedding::halfvec(1536) <=> "
        "(SELECT embedding::halfvec(1536) FROM transcript_chunks LIMIT 1) LIMIT 5"
    ),
    "technical_doc_chunks": (
        "EXPLAIN (ANALYZE, FORMAT JSON) SELECT id FROM technical_doc_chunks "
        "ORDER BY embedding::halfvec(1536) <=> "
        "(SELECT embedding::halfvec(1536) FROM technical_doc_chunks LIMIT 1) LIMIT 5"
    ),
}


def expected_index_name(table: str) -> str:
    """Nombre del índice que crea la 0005 y consulta ``ChunkStore.corpus_stats``."""
    return f"ix_{table}_embedding_hnsw"


def definition_is_aligned(indexdef: str | None) -> bool:
    """True si el índice usa la expresión halfvec y la operator class correcta.

    Un índice ``vector_cosine_ops`` (o sobre el vector completo) sería ignorado
    en silencio por una búsqueda halfvec: es el fallo que este chequeo ataja.
    """
    if not indexdef:
        return False
    normalized = indexdef.replace(" ", "").lower()
    return (
        f"halfvec({EMBEDDING_DIMENSIONS})" in normalized and EXPECTED_OPERATOR_CLASS in normalized
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica índices HNSW halfvec de la Sesión 11.",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Solo valida definiciones; no ejecuta EXPLAIN.",
    )
    parser.add_argument("--json", action="store_true", help="Salida JSON para automatizar.")
    return parser.parse_args()


async def _check_table(session: Any, table: str, *, explain: bool) -> dict[str, Any]:
    from sqlalchemy import text

    index_name = expected_index_name(table)
    indexdef = (await session.execute(text(_INDEX_SQL), {"index_name": index_name})).scalar()
    record: dict[str, Any] = {
        "table": table,
        "index": index_name,
        "exists": indexdef is not None,
        "aligned": definition_is_aligned(indexdef),
        "rows": 0,
        "plan": "no comprobado",
        "used": None,
    }
    if not explain or not record["exists"]:
        return record

    record["rows"] = int((await session.execute(text(_COUNT_SQL[table]))).scalar() or 0)
    if record["rows"] == 0:
        record["plan"] = "tabla vacía: EXPLAIN omitido"
        return record

    # Sesión de diagnóstico: no tocamos la configuración del servidor.
    await session.execute(text("SET enable_seqscan = off"))
    await session.execute(text("SET jit = off"))
    plan = (await session.execute(text(_EXPLAIN_SQL[table]))).scalar()
    if isinstance(plan, str):
        plan = json.loads(plan)
    record["used"] = index_name in json.dumps(plan)
    record["plan"] = f"Index Scan con {index_name}" if record["used"] else "NO usa el índice"
    return record


def _print_report(records: list[dict[str, Any]]) -> None:
    from scripts.rag_ops.common import format_table

    rows = [
        [
            str(record["table"]),
            str(record["rows"]),
            "sí" if record["exists"] else "NO",
            "sí" if record["aligned"] else "NO",
            "-" if record["used"] is None else ("sí" if record["used"] else "NO"),
            str(record["plan"]),
        ]
        for record in records
    ]
    print(
        format_table(
            ["colección", "filas", "existe", "alineado", "usado", "plan"],
            rows,
        )
    )


def _failures(records: list[dict[str, Any]], *, explain: bool) -> list[str]:
    problems: list[str] = []
    for record in records:
        table = record["table"]
        if not record["exists"]:
            problems.append(f"{table}: falta {record['index']}")
        elif not record["aligned"]:
            problems.append(
                f"{table}: {record['index']} no usa halfvec({EMBEDDING_DIMENSIONS}) "
                f"con {EXPECTED_OPERATOR_CLASS}"
            )
        elif explain and record["used"] is False:
            problems.append(f"{table}: el plan no usa {record['index']}")
    return problems


async def _main(args: argparse.Namespace) -> None:

    from app.foundation.persistence.database import get_async_session_factory
    from scripts.rag_ops.common import print_footer, print_header

    explain = not args.no_explain
    factory = get_async_session_factory()
    async with factory() as session:
        records = [await _check_table(session, table, explain=explain) for table in TABLES]

    if args.json:
        print(json.dumps({"tables": records}, indent=2, default=str))
    else:
        print_header("Índices HNSW halfvec (Sesión 11)")
        _print_report(records)
        print(
            "\nNota: en corpus pequeños el planner prefiere seq scan; el chequeo "
            "fuerza enable_seqscan=off para probar que el índice es utilizable."
        )
        print_footer("Migración esperada: 0005_session11_hnsw_multi_index.")

    problems = _failures(records, explain=explain)
    if problems:
        print("\nFALLOS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)

    print("\nOK: índices presentes, alineados y utilizables.")


if __name__ == "__main__":
    from scripts.rag_ops.common import run_async

    run_async(_main(_parse_args()))
