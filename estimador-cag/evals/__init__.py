"""Harness de evaluación del estimador conversacional.

Tres piezas:

- ``dataset.py``: registros ``GoldenCase`` tipados y un cargador JSON.
- ``metrics.py``: métricas deterministas (adherencia al esquema, cotas de coste,
  recall de contenido). El ``GEval`` basado en LLM de DeepEval es opt-in.
- ``run.py``: CLI que lanza cada caso contra uno de los dos modos (actor / acb)
  e imprime una tabla comparativa.

El ``golden_dataset.json`` preinstalado trae 16 casos que mezclan tipos de
proyecto, niveles de alcance, sabores NDA/regulatorios y un par de adversarios
fuera de alcance.
"""

from evals.dataset import GoldenCase, load_dataset
from evals.metrics import (
    ContentRecallMetric,
    CostBoundsMetric,
    MetricResult,
    SchemaAdherenceMetric,
    run_all_metrics,
)

__all__ = [
    "ContentRecallMetric",
    "CostBoundsMetric",
    "GoldenCase",
    "MetricResult",
    "SchemaAdherenceMetric",
    "load_dataset",
    "run_all_metrics",
]
