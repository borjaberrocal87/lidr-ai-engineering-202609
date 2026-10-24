#!/usr/bin/env python3
"""Smoke test de búsqueda semántica contra el corpus persistido (Sesión 08).

Reemplaza el rol de ``compare.py``: en vez de medir similitud entre dos textos
sueltos, ejercita la ruta real —HTTP contra ``POST /api/v1/embeddings/ingest``
y ``POST /api/v1/search``— con cinco queries que prueban el corpus desde
ángulos distintos (match directo, reformulación semántica, dominio ajeno,
ambigua y muy específica).

Idempotente: primero ingesta ``data/budgets_sample.json`` (un documento por
presupuesto); los documentos ya persistidos responden 409 y se saltan, así que
reejecutar el script nunca duplica datos.

Uso::

    # stack arriba primero: docker compose up -d --build api
    docker compose run --rm api python scripts/query_examples.py \
        | tee data/output/session08/output_examples.txt

    # o desde el host (con la API en localhost:8000):
    uv run python scripts/query_examples.py

La URL base se toma de ``ESTIMATOR_BASE_URL`` si existe; si no, el script
prueba ``http://localhost:8000`` y ``http://api:8000`` (alias de Compose).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "budgets_sample.json"

CANDIDATE_BASE_URLS = ("http://localhost:8000", "http://api:8000")

QUERIES: list[tuple[str, str]] = [
    (
        "Componente directo conocido (sanity check)",
        "REST API development with JWT authentication for financial sector",
    ),
    (
        "Reformulación semántica (mismo concepto, otro vocabulario)",
        "secure backend service with token-based access control for banking applications",
    ),
    (
        "Dominio distinto (no debería estar en el corpus)",
        "mobile application for restaurant reservations",
    ),
    (
        "Consulta ambigua (sin match dominante)",
        "integration with external system",
    ),
    (
        "Consulta muy específica (vocabulario técnico preciso)",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
]

TOP_K = 5
CONTENT_PREVIEW_CHARS = 120


def resolve_base_url(client: httpx.Client) -> str:
    """Respeta ``ESTIMATOR_BASE_URL`` o prueba los candidatos habituales."""
    explicit = os.environ.get("ESTIMATOR_BASE_URL")
    candidates = (explicit,) if explicit else CANDIDATE_BASE_URLS
    for base_url in candidates:
        try:
            if client.get(f"{base_url}/health").status_code == 200:
                return base_url
        except httpx.TransportError:
            continue
    print(
        "ERROR: no hay API del estimador accesible. Levanta el stack "
        "(docker compose up -d --build api) o define ESTIMATOR_BASE_URL.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def ingest_corpus(client: httpx.Client, base_url: str) -> None:
    """Un documento por presupuesto; 409 = ya ingerido (idempotente)."""
    budgets = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    created, skipped = 0, 0
    for budget in budgets:
        response = client.post(
            f"{base_url}/api/v1/embeddings/ingest",
            json={
                "source_path": f"data/budgets_sample.json::{budget['budget_id']}",
                "document_type": "historical_budget",
                "content": budget,
            },
        )
        if response.status_code == 200:
            created += 1
        elif response.status_code == 409:
            skipped += 1
        else:
            print(
                f"ERROR ingiriendo {budget['budget_id']}: "
                f"{response.status_code} {response.text[:200]}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    print(f"Corpus: {len(budgets)} presupuestos — {created} ingeridos, {skipped} ya presentes.")


def run_queries(client: httpx.Client, base_url: str) -> None:
    for index, (label, query) in enumerate(QUERIES, start=1):
        response = client.post(
            f"{base_url}/api/v1/search",
            json={"query": query, "k": TOP_K},
        )
        response.raise_for_status()
        body = response.json()

        print()
        print(f"[{index}/5] {label}")
        print(f'    query: "{query}"')
        print(f"    search_time_ms: {body['search_time_ms']}")
        print(f"    {'chunk_id':>8}  {'distance':>8}  {'chunk_type':<18}  content")
        for hit in body["results"]:
            preview = " ".join(hit["content"].split())[:CONTENT_PREVIEW_CHARS]
            print(
                f"    {hit['chunk_id']:>8}  {hit['distance']:>8.4f}  "
                f"{hit['chunk_type']:<18}  {preview}"
            )


def main() -> int:
    with httpx.Client(timeout=120.0) as client:
        base_url = resolve_base_url(client)
        print(f"Estimator API: {base_url}")
        ingest_corpus(client, base_url)
        run_queries(client, base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
