#!/usr/bin/env python3
"""Demo del subbloque 2.5 — pseudonimización con Presidio (español) + tabla de mappings.

Forma ejecutable de la demo en vivo (sin escribir en el REPL). Construye
``build_analyzer()`` conectado a ``es_core_news_md`` + los reconocedores
personalizados, y luego ejecuta un ``ConsistentPseudonymizer`` sobre una
transcripción real e imprime el resultado.

Por qué este script NO ejecuta el contraste "el inglés falla": forzar el motor
inglés por defecto dispara una descarga de ``en_core_web_lg`` de ~382 MB y,
dentro de docker (donde solo está instalado el modelo español), provocaría un
crash. Ese fallo es un punto didáctico hablado — su justificación está
documentada en ``app/ingestion/pii/analyzer.py``. Aquí mostramos el *arreglo*
funcionando.

Usage::

    uv run python scripts/demo_pii_s06.py
    # or, inside docker:
    docker compose exec estimator python scripts/demo_pii_s06.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Se requiere un salt para construir el pseudonimizador; usa uno de demo si no está definido.
os.environ.setdefault("PSEUDONYM_HASH_SALT", "demo-salt")

from app.ingestion.pii import (  # noqa: E402
    ConsistentPseudonymizer,
    InMemoryMappingStore,
    build_analyzer,
)

TRANSCRIPT = ROOT / "data" / "seed" / "transcripts" / "transcripcion_2025-02-03_betanorte.txt"


def main() -> int:
    if not TRANSCRIPT.exists():
        print(f"ERROR: no existe {TRANSCRIPT}")
        return 1

    pseudo = ConsistentPseudonymizer(
        analyzer=build_analyzer(),  # es_core_news_md + reconocedores BUDGET_ID/CLIENT_CODE
        mapping_store=InMemoryMappingStore(),
        salt=os.environ["PSEUDONYM_HASH_SALT"],
    )
    result = pseudo.pseudonymize(TRANSCRIPT.read_text(encoding="utf-8"))

    print(f"Transcripción: {TRANSCRIPT.name}")
    print("\n--- Texto pseudonimizado (primeros 400 chars) ---")
    print(result.pseudonymized_text[:400])

    print(f"\n--- Mappings aplicados ({len(result.applied)}) ---")
    for m in result.applied[:12]:
        print(f"  {m.entity_type:14} → {m.pseudonym}")
    print(
        "\nMismo valor original → mismo pseudónimo en todo el documento "
        "(consistencia por valor, no por chunk)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
