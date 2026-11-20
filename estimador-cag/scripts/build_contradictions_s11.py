#!/usr/bin/env python3
"""Ingesta el fixture de fuentes contradictorias de la Sesión 11 (40h vs 90h).

El consenso de horas por tarea (Sesión 10) promedia los análogos más cercanos en
un único número. La Sesión 11 muestra lo que eso esconde: cuando los análogos
discrepan de verdad, el promedio miente. Este fixture planta dos proyectos
históricos que comparten el nombre de tarea "Real-time notifications delivery
service" pero con alcances opuestos: un email-only ligero (40h) y una plataforma
multicanal regulada (90h). Una tarea del wizard que casa con ambos dispara la
dispersión de vecinos, y la síntesis saca a la luz un RANGO 40-90h con su razón
en vez de un punto medio de ~65h que aparenta seguridad.

Se ingesta con las MISMAS etiquetas que el corpus base de tareas
(``document_type='historical_task_breakdown'`` / ``chunk_type='historical_task'``),
así que lo recupera la misma búsqueda filtrada y convive con él. Es idempotente
(409 = ya presente). Reutiliza los ayudantes del builder base; no añade HTTP nuevo.

Uso::

    docker compose exec estimator python scripts/build_contradictions_s11.py
    uv run python scripts/build_contradictions_s11.py            # host, API en :8000

Para borrarlo (o vaciar todo el corpus de tareas)::

    DELETE FROM documents WHERE document_type = 'historical_task_breakdown';
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_task_corpus import ingest_corpus  # noqa: E402

FIXTURE_PATH = ROOT / "data" / "task_corpus_contradictions.json"


def main() -> None:
    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    print(f"Cargados {len(corpus)} proyectos contradictorios desde {FIXTURE_PATH.name}.")
    ingest_corpus(corpus)


if __name__ == "__main__":
    main()
