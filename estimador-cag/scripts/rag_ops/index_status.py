#!/usr/bin/env python3
"""Audita el estado y el uso de los índices de la tabla ``chunks``.

Responde a la pregunta operativa que aparece cuando algo va lento: ¿existe el
índice, cuánto ocupa y lo está usando realmente el planner? Combina
``pg_stat_user_indexes`` con ``pg_am`` y las operator classes de cada índice.

Uso::

    docker compose exec api python -m scripts.rag_ops.index_status
    docker compose exec api python -m scripts.rag_ops.index_status --json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_async_session_factory
from scripts.rag_ops.common import (
    format_table,
    print_footer,
    print_header,
    run_async,
)

INDEX_SQL = text(
    """
    SELECT
        i.indexrelname AS index_name,
        am.amname AS index_type,
        pg_size_pretty(pg_relation_size(i.indexrelid)) AS size,
        i.idx_scan AS scans,
        COALESCE(i.last_idx_scan::text, 'nunca') AS last_used
    FROM pg_stat_user_indexes i
    JOIN pg_class c ON c.oid = i.indexrelid
    JOIN pg_am am ON am.oid = c.relam
    WHERE i.relname = 'chunks'
    ORDER BY pg_relation_size(i.indexrelid) DESC
    """
)

OPERATOR_CLASS_SQL = text(
    """
    SELECT
        i.indexrelname AS index_name,
        string_agg(DISTINCT op.opcname, ', ' ORDER BY op.opcname) AS operator_classes
    FROM pg_stat_user_indexes i
    JOIN pg_index pgi ON pgi.indexrelid = i.indexrelid
    JOIN pg_opclass op ON op.oid = ANY (pgi.indclass::oid[])
    WHERE i.relname = 'chunks'
    GROUP BY i.indexrelname
    """
)

TABLE_SQL = text(
    """
    SELECT
        pg_size_pretty(pg_table_size('chunks')) AS table_size,
        count(*) AS chunk_count
    FROM chunks
    """
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estado, tamaño y uso de los índices de chunks.",
    )
    parser.add_argument("--json", action="store_true", help="Salida JSON para automatizar.")
    return parser.parse_args()


async def _collect(session: AsyncSession) -> dict[str, Any]:
    indexes = [dict(row) for row in (await session.execute(INDEX_SQL)).mappings().all()]
    classes = {
        row["index_name"]: row["operator_classes"]
        for row in (await session.execute(OPERATOR_CLASS_SQL)).mappings().all()
    }
    for index in indexes:
        index["operator_classes"] = classes.get(index["index_name"], "—")

    table_row = (await session.execute(TABLE_SQL)).mappings().one()
    return {
        "table": dict(table_row),
        "indexes": indexes,
    }


def _print_report(report: dict[str, Any]) -> None:
    table = report["table"]
    indexes = report["indexes"]

    print(f"Tabla chunks: {table['chunk_count']} filas · {table['table_size']}\n")
    if not indexes:
        print("No hay índices registrados sobre chunks.")
        return

    rows = [
        [
            str(index["index_name"]),
            str(index["index_type"]),
            str(index["size"]),
            str(index["scans"]),
            str(index["last_used"]),
            str(index["operator_classes"]),
        ]
        for index in indexes
    ]
    print(
        format_table(
            ["índice", "tipo", "tamaño", "scans", "último uso", "operator class"],
            rows,
        )
    )


def _print_warnings(report: dict[str, Any]) -> None:
    indexes = report["indexes"]
    vector_indexes = [index for index in indexes if str(index["index_type"]) in {"hnsw", "ivfflat"}]
    if not vector_indexes:
        print(
            "\nAviso: no hay índice vectorial. El retrieval actual es sequential "
            "scan; crea HNSW con scripts/rag_ops/sql/01_hnsw.sql."
        )
        return

    unused = [index["index_name"] for index in vector_indexes if int(index["scans"]) == 0]
    if unused:
        print(
            "\nAviso: índice(s) vectorial(es) sin ningún scan: "
            f"{', '.join(str(name) for name in unused)}. "
            "Comprueba que el operador de la query coincide con la operator class "
            "(04_monitorizacion.sql)."
        )


async def _main(args: argparse.Namespace) -> None:
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        report = await _collect(session)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    print_header("Estado de índices del store vectorial")
    _print_report(report)
    _print_warnings(report)
    print_footer("Ejecuta este informe antes y después de cada decisión de índice.")


if __name__ == "__main__":
    run_async(_main(_parse_args()))
