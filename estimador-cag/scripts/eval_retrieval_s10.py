#!/usr/bin/env python3
"""Mide la calidad de la recuperación multifuente en configuraciones nombradas.

Ejecuta el golden set (``evals/golden_retrieval.json``) por el pipeline
multifuente y reporta precisión@k (media sobre las consultas), latencia
extremo a extremo (media sobre las repeticiones medidas) y el desglose por
consulta.

Las configuraciones son DATOS, no ramas de código: cada una es un ``StageConfig``
que enciende o apaga etapas concretas (las cuatro base A-D del pre-work, luego
enrutado, plan de consulta, vigencia y el pipeline completo). Añadir una fila es
añadir un ``NamedConfig``.

Notas de método:
* El umbral de distancia es permisivo para que el top-k no se trunque por el
  suelo de relevancia: comparamos CALIDAD DE RANKING, no el soft-fail.
* Un chunk recuperado es relevante si su ``source_id`` (budget_id / transcript_id
  / doc_id) está en ``relevant_source_ids``; ``relevant_budget_ids`` se sigue
  aceptando para las consultas originales de solo presupuestos.
* La latencia es extremo a extremo: incluye el embedding por sub-consulta y, en
  las filas de enrutado/plan de consulta, las llamadas LLM que esas técnicas
  añaden. Ese sobrecoste ES la contrapartida que pondera la matriz de decisión.

Uso (host, stack arriba y las tres fuentes ingestadas + OPEN_AI_KEY):

    uv run python scripts/eval_retrieval_s10.py

Ingesta previa: ``scripts/query_examples.py`` (presupuestos) +
``scripts/build_multi_index_corpus.py`` (transcripciones y documentación).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.dependencies import (  # noqa: E402
    get_async_session_factory,
    get_chunk_store,
    get_llm_wrapper,
    get_reranker,
)
from app.generation.rag.retrieval.multi_source import (  # noqa: E402
    StageConfig,
    retrieve_multi_source,
)
from app.generation.rag.retrieval.sources import Collection  # noqa: E402
from app.generation.rag.schemas import RetrievedChunk  # noqa: E402
from scripts.rag_ops.common import require_embedder  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"

# Sin suelo de relevancia: queremos un top-k completo para evaluar ranking.
NO_FLOOR_THRESHOLD = 2.0
# Muestras de latencia por (consulta, configuración): un warm-up descartado + N.
MEASURED_RUNS = 3
# Solo presupuestos para las filas A-D/E/G (comparables con el pre-work).
BUDGET_ONLY = [Collection.BUDGET]


@dataclass(frozen=True)
class NamedConfig:
    """Una configuración medida: etiqueta + etapas + ámbito de enrutado."""

    id: str
    label: str
    stages: StageConfig
    explicit: list[Collection] | None  # None = deja decidir al enrutador


def _stages(**overrides) -> StageConfig:
    """StageConfig con los defaults del arnés (umbral permisivo, k=5)."""
    base = {
        "routing_enabled": False,
        "query_plan_enabled": False,
        "search_mode": "vector",
        "rerank": False,
        "freshness_enabled": False,
        "top_k": 5,
        "distance_threshold": NO_FLOOR_THRESHOLD,
    }
    base.update(overrides)
    return StageConfig(**base)


# Las configuraciones nombradas (DATOS). A-D reproducen el pre-work sobre
# presupuestos; E-H añaden las técnicas del directo.
CONFIGS: list[NamedConfig] = [
    NamedConfig("A", "Vector", _stages(search_mode="vector"), BUDGET_ONLY),
    NamedConfig("B", "Híbrida", _stages(search_mode="hybrid"), BUDGET_ONLY),
    NamedConfig("C", "Vector+Rerank", _stages(search_mode="vector", rerank=True), BUDGET_ONLY),
    NamedConfig("D", "Híbrida+Rerank", _stages(search_mode="hybrid", rerank=True), BUDGET_ONLY),
    NamedConfig(
        "E",
        "Híbrida+Rerank+Vigencia",
        _stages(search_mode="hybrid", rerank=True, freshness_enabled=True),
        BUDGET_ONLY,
    ),
    NamedConfig(
        "F",
        "Enrutado multifuente",
        _stages(search_mode="hybrid", rerank=True, routing_enabled=True),
        None,
    ),
    NamedConfig(
        "G",
        "Plan de consulta",
        _stages(search_mode="hybrid", rerank=True, query_plan_enabled=True),
        BUDGET_ONLY,
    ),
    NamedConfig(
        "H",
        "Pipeline completo",
        _stages(
            search_mode="hybrid",
            rerank=True,
            routing_enabled=True,
            query_plan_enabled=True,
            freshness_enabled=True,
        ),
        None,
    ),
]


def relevant_ids(query: dict) -> set[str]:
    """Ids anotados (ids genéricos de fuente; fallback a budget ids)."""
    return set(query.get("relevant_source_ids") or query.get("relevant_budget_ids") or [])


def precision_at_k(chunks: list[RetrievedChunk], relevant: set[str], k: int) -> float:
    """Fracción del top-k cuyo documento fuente es genuinamente relevante."""
    top = chunks[:k]
    hits = sum(1 for chunk in top if (chunk.source_id or chunk.budget_id) in relevant)
    return hits / k


async def _run_once(
    cfg: NamedConfig,
    query_text: str,
    *,
    embedder,
    store,
    session_factory,
    reranker,
    llm_wrapper,
    reference_date: date,
):
    outcome = await retrieve_multi_source(
        query_text=query_text,
        embedder=embedder,
        store=store,
        session_factory=session_factory,
        stages=cfg.stages,
        reference_date=reference_date,
        llm_wrapper=llm_wrapper,
        reranker=reranker,
        explicit_collections=cfg.explicit,
    )
    return outcome.chunks


def _parse_args() -> argparse.Namespace:
    """Permite un smoke barato (1 query, 1 config) antes del barrido completo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="Limita el número de consultas (por defecto, todas).",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Lista de configuraciones separadas por comas, p. ej. A,D,H.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=MEASURED_RUNS,
        help=f"Repeticiones medidas por consulta/config (por defecto {MEASURED_RUNS}).",
    )
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()
    get_settings()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]
    if args.queries is not None:
        queries = queries[: max(0, args.queries)]
    selected = CONFIGS
    if args.configs:
        wanted = {part.strip().upper() for part in args.configs.split(",") if part.strip()}
        selected = [cfg for cfg in CONFIGS if cfg.id in wanted]
        unknown = wanted - {cfg.id for cfg in CONFIGS}
        if unknown:
            print(f"Configuraciones desconocidas: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
    if not selected:
        print("No hay configuraciones seleccionadas.", file=sys.stderr)
        return 2
    k = int(golden.get("k", 5))
    reference_date = date.today()

    embedder = require_embedder()
    store = get_chunk_store()
    session_factory = get_async_session_factory()
    reranker = get_reranker()
    llm_wrapper = get_llm_wrapper()

    results: dict[str, list[float]] = {cfg.id: [] for cfg in selected}
    latencies: dict[str, list[float]] = {cfg.id: [] for cfg in selected}
    per_query: dict[str, dict[str, float]] = {cfg.id: {} for cfg in selected}

    print(
        f"Golden set: {len(queries)} consultas · k={k} · "
        f"configuraciones {','.join(cfg.id for cfg in selected)}"
    )
    for query in queries:
        relevant = relevant_ids(query)
        for cfg in selected:
            # Warm-up descartado (carga de modelo, cachés de conexión…).
            await _run_once(
                cfg,
                query["query"],
                embedder=embedder,
                store=store,
                session_factory=session_factory,
                reranker=reranker,
                llm_wrapper=llm_wrapper,
                reference_date=reference_date,
            )
            samples: list[float] = []
            chunks: list[RetrievedChunk] = []
            for _ in range(args.runs):
                started = time.perf_counter()
                chunks = await _run_once(
                    cfg,
                    query["query"],
                    embedder=embedder,
                    store=store,
                    session_factory=session_factory,
                    reranker=reranker,
                    llm_wrapper=llm_wrapper,
                    reference_date=reference_date,
                )
                samples.append((time.perf_counter() - started) * 1000)
            precision = precision_at_k(chunks, relevant, k)
            results[cfg.id].append(precision)
            latencies[cfg.id].append(statistics.fmean(samples))
            per_query[cfg.id][query["id"]] = precision

    print("\n=== Precisión@k y latencia por configuración ===")
    header = f"{'ID':<3} {'Configuración':<28} {'Precisión@' + str(k):>12} {'Latencia ms':>12}"
    print(header)
    print("-" * len(header))
    for cfg in selected:
        precision = statistics.fmean(results[cfg.id]) if results[cfg.id] else 0.0
        latency = statistics.fmean(latencies[cfg.id]) if latencies[cfg.id] else 0.0
        print(f"{cfg.id:<3} {cfg.label:<28} {precision:>12.3f} {latency:>12.0f}")

    print("\n=== Precisión por consulta ===")
    query_ids = [query["id"] for query in queries]
    print(f"{'Config':<6} " + " ".join(f"{qid:>6}" for qid in query_ids))
    for cfg in selected:
        row = " ".join(f"{per_query[cfg.id].get(qid, 0.0):>6.2f}" for qid in query_ids)
        print(f"{cfg.id:<6} {row}")

    print(
        "\nMatriz de decisión: A-D son la base; E mide vigencia, F enrutado, "
        "G plan de consulta y H el pipeline completo. Compara precisión contra "
        "latencia: cada etapa cara debe ganarse su sitio."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
