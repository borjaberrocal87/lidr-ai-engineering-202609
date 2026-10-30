"""Piezas compartidas de RAG Ops.

Aquí viven las cinco consultas de referencia, el formateo de tablas y las
estadísticas de latencia. Los scripts de benchmark no duplican esta lógica y
comparten exactamente las mismas queries, de modo que sus números son
comparables entre sí.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from collections.abc import Coroutine, Sequence
from typing import Any

from app.dependencies import get_embedder
from app.embedding_pipeline.embedder import OpenAIEmbedder
from scripts.query_examples import QUERIES

# Resultado de embeber una consulta de referencia: etiqueta, texto y vector.
EmbeddedQuery = tuple[str, str, list[float]]

# Top-k que usan los benchmarks y el endpoint /search.
DEFAULT_K = 5
# Top-k de la verdad de fondo para calcular recall a 5.
GROUND_TRUTH_K = 10


def print_header(title: str) -> None:
    """Cabecera común para que todos los scripts se reconozcan como RAG Ops."""
    print(f"\n=== RAG Ops · {title} ===")


def print_footer(message: str) -> None:
    print(f"\n{message}")


def require_embedder() -> OpenAIEmbedder:
    """Devuelve el embedder del proyecto o sale con un mensaje accionable."""
    embedder = get_embedder()
    if embedder is None:
        print(
            "ERROR: no hay cliente de embeddings configurado.\n"
            "Define OPEN_AI_KEY (o EMBEDDING_BASE_URL + EMBEDDING_API_KEY) "
            "en estimador-cag/.env y recrea el contenedor.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return embedder


def embed_queries(embedder: OpenAIEmbedder) -> list[EmbeddedQuery]:
    """Embebe las consultas una sola vez, fuera de las mediciones de latencia.

    El tiempo de ida y vuelta a OpenAI no depende del índice de Postgres, así
    que pagarlo una vez evita contaminar el benchmark SQL.
    """
    print(f"Preparando {len(QUERIES)} consultas de referencia...")
    return [(label, query, embedder.embed_one(query)) for label, query in QUERIES]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Tabla de texto alineada, sin dependencias externas."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return f"{header_line}\n{separator}\n{body}"


def truncate(text: str, width: int) -> str:
    """Recorta con elipsis para que las tablas sigan siendo legibles."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def latency_summary(samples_ms: Sequence[float]) -> str:
    """Resumen global de latencias: media, mediana y p95 cuando hay muestra."""
    mean = statistics.fmean(samples_ms)
    median = statistics.median(samples_ms)
    parts = [f"media={mean:.1f} ms", f"mediana={median:.1f} ms"]
    if len(samples_ms) >= 20:
        p95 = statistics.quantiles(samples_ms, n=20)[-1]
        parts.append(f"p95={p95:.1f} ms")
    return " | ".join(parts)


def latency_stats(samples_ms: Sequence[float]) -> tuple[float, float]:
    """Media y desviación típica poblacional de una muestra de latencias."""
    return statistics.fmean(samples_ms), statistics.pstdev(samples_ms)


def vector_literal(vector: Sequence[float]) -> str:
    """Vector en el formato que espera pgvector: ``[0.1,0.2,...]``."""
    return "[" + ",".join(repr(component) for component in vector) + "]"


class Stopwatch:
    """Cronómetro de pared para medir una operación con ``with``."""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def run_async(main: Coroutine[Any, Any, None]) -> None:
    """Ejecuta el ``main`` async de un script con errores legibles."""
    try:
        asyncio.run(main)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:  # frontera CLI: traducimos el error a salida legible
        print(
            f"ERROR: {type(exc).__name__}: {exc}\n"
            "Revisa que Postgres esté levantado y que DATABASE_URL apunte al "
            "servicio correcto.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
