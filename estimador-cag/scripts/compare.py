#!/usr/bin/env python3
"""Sanity check de embeddings — similitud coseno entre dos textos.

Embebe dos textos con ``text-embedding-3-small`` (vía ``OpenAIEmbedder``) e
imprime su similitud coseno. El coseno se calcula con la stdlib: ``math.sumprod``
para el producto escalar y ``math.hypot`` para las normas L2. Sin numpy.

Uso::

    # fuera del contenedor (con el .env cargado):
    uv run python scripts/compare.py \\
        --text-a "OAuth 2.0 authentication backend for fintech" \\
        --text-b "JWT-based authorization service for banking app"

    # dentro del contenedor:
    docker compose exec api python scripts/compare.py \\
        --text-a "..." --text-b "..."
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Hace importable ``app`` cuando el script se ejecuta directamente desde ``scripts/``.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Producto escalar dividido por el producto de las normas L2."""
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return math.sumprod(a, b) / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Similitud coseno entre dos textos embebidos.")
    parser.add_argument("--text-a", required=True, help="Primer texto.")
    parser.add_argument("--text-b", required=True, help="Segundo texto.")
    args = parser.parse_args()

    embedder = OpenAIEmbedder.from_settings(get_settings())
    if embedder is None:
        print(
            "ERROR: no hay embeddings configurados "
            "(OPEN_AI_KEY o EMBEDDING_BASE_URL + EMBEDDING_API_KEY en .env).",
            file=sys.stderr,
        )
        return 1

    similarity = cosine_similarity(
        embedder.embed_one(args.text_a),
        embedder.embed_one(args.text_b),
    )

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
