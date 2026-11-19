#!/usr/bin/env python3
"""Evaluación determinista de la calidad de generación (Sesión 11, directo).

La sesión pre-work dejó un baseline RAGAS (``evals/ragas_baseline_s11.json``). Este
arnés NO lo sustituye: mide, sin jueces RAGAS y sin red extra, las señales que las
técnicas del directo cambian, para poder hacer *medir → conmutar → re-medir*:

* **Citas**: tasa de líneas con cita real y citas colgantes (verificación S11).
* **Semántica**: tasa de líneas respaldadas por la evidencia y tasa degradada
  (gate de alucinaciones).
* **Contexto**: tokens enviados al generador con y sin augmentación.
* **Síntesis**: rango 40-90h que emite el fixture de contradicciones (demo).

Cada configuración con nombre (``baseline``, ``gate``, ``augment``, ``full``) fija
las palancas de forma EXPLÍCITA, sin depender de overrides de Redis: el resultado
es reproducible en host o contenedor.

Uso::

    uv run python scripts/eval_quality_s11.py --config full --limit 2
    uv run python scripts/eval_quality_s11.py --compare --output /tmp/s11_quality.json
    uv run python scripts/eval_quality_s11.py --config full --gate

Prerrequisitos: ``.env`` con clave de embeddings/LLM y el corpus de presupuestos
ingerido (``uv run python scripts/query_examples.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_PATH = ROOT / "evals" / "golden_generation_s11.json"
CONTRADICTIONS_PATH = ROOT / "data" / "task_corpus_contradictions.json"

# Umbral permissivo para que el generador SIEMPRE vea contexto: medimos
# generación y gate, no el soft-fail del retrieval.
GENERATION_THRESHOLD = 1.2

# Puertas por defecto del modo --gate (regresión, no excelencia).
MIN_CITATION_GROUNDING = 0.8
MAX_DEGRADED_RATE = 0.2


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Palancas explícitas de una configuración con nombre."""

    id: str
    label: str
    gate: bool
    augment: bool


CONFIGS: dict[str, QualityConfig] = {
    "baseline": QualityConfig("baseline", "Baseline (pipeline S10)", gate=False, augment=False),
    "gate": QualityConfig("gate", "+ Gate semántico", gate=True, augment=False),
    "augment": QualityConfig("augment", "+ Augmentación", gate=False, augment=True),
    "full": QualityConfig("full", "+ Gate y augmentación", gate=True, augment=True),
}
ALL_CONFIG_IDS = tuple(CONFIGS)


def summarize(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    """Agrega una lista de muestras en las métricas del scoreboard.

    Cada muestra trae ``citation_report`` y, opcionalmente, ``hallucination_report``
    (dicts serializados). La tasa de citas es siempre calculable; la semántica solo
    cuando el gate corrió (si no, se deja a 0 y cuenta como "no medida").
    """
    total_lines = verified = dangling = 0
    gate_total = gate_grounded = gate_degraded = 0
    context_tokens = 0

    for sample in samples:
        citation = sample.get("citation_report") or {}
        total_lines += int(citation.get("total_lines", 0))
        verified += int(citation.get("grounded_lines", 0))
        dangling += len(citation.get("dangling_citations") or [])

        gate = sample.get("hallucination_report")
        if gate:
            gate_total += int(gate.get("total_lines", 0))
            gate_grounded += int(gate.get("grounded_lines", 0))
            gate_degraded += int(gate.get("degraded_lines", 0))

        context_tokens += int(sample.get("context_tokens", 0))

    return {
        "samples": len(samples),
        "lines": total_lines,
        "citation_grounding_rate": (verified / total_lines) if total_lines else 0.0,
        "dangling_citations": dangling,
        "semantic_lines": gate_total,
        "semantic_grounding_rate": (gate_grounded / gate_total) if gate_total else 0.0,
        "degraded_rate": (gate_degraded / gate_total) if gate_total else 0.0,
        "avg_context_tokens": (context_tokens / len(samples)) if samples else 0.0,
    }


async def synthesis_demo() -> dict[str, Any]:
    """Rango que la síntesis emite sobre el fixture 40h vs 90h (determinista).

    No necesita BD ni LLM: calcula la dispersión real del consenso y deja que la
    síntesis decida. Sirve para verificar que el desacuerdo no se esconde.
    """
    from app.generation.rag.quality.synthesis import synthesize_hour_range
    from app.generation.rag.schemas import TaskNeighbor
    from app.generation.rag.task_hours import _consensus

    corpus = json.loads(CONTRADICTIONS_PATH.read_text(encoding="utf-8"))
    hours = sorted(
        component["estimated_hours"]
        for project in corpus
        for component in project["components"]
        if component["name"] == "Real-time notifications delivery service"
    )
    neighbors = [
        TaskNeighbor(source_id=index, estimated_hours=value, distance=0.10)
        for index, value in enumerate(hours, start=1)
    ]
    _hours, _reliability, dispersion = _consensus([(value, 0.10) for value in hours])
    result = await synthesize_hour_range(neighbors, dispersion, threshold=0.35)
    return {
        "neighbour_hours": hours,
        "dispersion": dispersion,
        "range": None if result is None else [result.low_hours, result.high_hours],
        "reason": None if result is None else result.reason,
    }


async def _collect(
    item: dict[str, Any], config: QualityConfig, retriever: Any, encoder: Any
) -> dict[str, Any]:
    """Ejecuta el pipeline real para un ítem bajo una configuración explícita."""
    from app.config import settings
    from app.dependencies import get_llm_wrapper
    from app.generation.rag.brief_extractor import compose_retrieval_text, extract_project_brief
    from app.generation.rag.context_builder import assemble_context, fit_chunks_to_budget
    from app.generation.rag.grounded_estimator import generate_grounded_estimate
    from app.generation.rag.grounding import verify_citations_for_chunks
    from app.generation.rag.quality.augmentation import AugmentationPolicy, apply_augmentation
    from app.generation.rag.quality.hallucination import HallucinationGate

    transcript = item.get("transcript") or item["query"]
    brief = await extract_project_brief(transcript)
    retrieval = await retriever.retrieve_chunks(
        query_text=compose_retrieval_text(brief),
        search_mode="vector",
        rerank=False,
        top_k=settings.rag_retrieval_top_k,
        recall_k=settings.retrieval_recall_top_k,
        rerank_top_n=settings.rerank_top_n,
        distance_threshold=GENERATION_THRESHOLD,
        rrf_k=settings.rrf_k,
        chunk_types=["budget_component"],
    )
    kept = fit_chunks_to_budget(retrieval.chunks, settings.rag_max_context_tokens, encoder)
    if config.augment:
        kept = apply_augmentation(kept, AugmentationPolicy(compress=True, reorder=True))
    context_block = assemble_context(kept)
    estimate = await generate_grounded_estimate(context_block, brief)
    citation = verify_citations_for_chunks(estimate, kept)

    hallucination = None
    if config.gate:
        gate = HallucinationGate(
            judge=get_llm_wrapper(),
            model=settings.hallucination_judge_model,
            tolerance=settings.hallucination_numeric_tolerance,
        )
        hallucination = (await gate.audit(estimate, kept)).model_dump()

    return {
        "id": item["id"],
        "config": config.id,
        "context_tokens": len(encoder.encode(context_block)),
        "citation_report": citation.model_dump(),
        "hallucination_report": hallucination,
    }


async def _run_config(
    config: QualityConfig, items: list[dict[str, Any]], limit: int | None
) -> dict[str, Any]:
    from app.dependencies import get_semantic_retriever, get_token_encoder

    retriever = get_semantic_retriever()
    if retriever is None:
        raise SystemExit("Embedding service is not configured (OPEN_AI_KEY?).")
    encoder = get_token_encoder()
    selected = items[:limit] if limit is not None else items
    samples = []
    for item in selected:
        print(f"[{config.id}] {item['id']} ...")
        samples.append(await _collect(item, config, retriever, encoder))
    return {"config": config.id, "label": config.label, "summary": summarize(samples)}


def _gate_failures(summary: dict[str, float | int]) -> list[str]:
    """Violaciones de la puerta de regresión para un resumen."""
    failures: list[str] = []
    if int(summary["dangling_citations"]) > 0:
        failures.append(f"citas colgantes: {summary['dangling_citations']}")
    if float(summary["citation_grounding_rate"]) < MIN_CITATION_GROUNDING:
        failures.append(
            f"citacion {summary['citation_grounding_rate']:.2f} < {MIN_CITATION_GROUNDING}"
        )
    if int(summary["semantic_lines"]) > 0 and float(summary["degraded_rate"]) > MAX_DEGRADED_RATE:
        failures.append(f"degradadas {summary['degraded_rate']:.2f} > {MAX_DEGRADED_RATE}")
    return failures


def _print_scoreboard(report: dict[str, Any]) -> None:
    print("\nconfig      citas  colg  semantica  degradadas  tokens")
    for entry in report["configs"]:
        s = entry["summary"]
        print(
            f"{entry['config']:<11} {s['citation_grounding_rate']:>5.2f} "
            f"{s['dangling_citations']:>5} {s['semantic_grounding_rate']:>10.2f} "
            f"{s['degraded_rate']:>11.2f} {s['avg_context_tokens']:>7.0f}"
        )
    demo = report.get("synthesis_demo") or {}
    if demo.get("range"):
        print(f"síntesis: vecinos {demo['neighbour_hours']}h → rango {demo['range']}")


async def _run(args: argparse.Namespace) -> int:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = golden["queries"]
    config_ids = ALL_CONFIG_IDS if args.compare else (args.config,)

    entries = []
    for config_id in config_ids:
        entries.append(await _run_config(CONFIGS[config_id], items, args.limit))

    report: dict[str, Any] = {
        "golden": str(GOLDEN_PATH.name),
        "synthesis_demo": await synthesis_demo(),
        "configs": entries,
    }

    _print_scoreboard(report)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nInforme escrito en {args.output}")

    if args.gate:
        failures = [
            f"{entry['config']}: {failure}"
            for entry in entries
            for failure in _gate_failures(entry["summary"])
        ]
        if failures:
            print("\nGATE: FAIL")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\nGATE: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=ALL_CONFIG_IDS, default="full")
    parser.add_argument("--compare", action="store_true", help="Ejecuta las 4 configuraciones.")
    parser.add_argument("--limit", type=int, default=None, help="Máximo de ítems del golden.")
    parser.add_argument("--gate", action="store_true", help="Falla (exit 1) si hay regresión.")
    parser.add_argument("--output", type=Path, default=None, help="Escribe el informe JSON.")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
