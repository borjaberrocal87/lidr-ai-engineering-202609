"""Stress test del CAG (sesión 6).

Subpaquete de evals que instrumenta y somete el sistema CAG de la sesión 5 a tres
escenarios de carga (multi-turno largo, adjuntos grandes, contradicciones) para
producir el baseline cuantitativo del que parte RAG en el directo.

Piezas:

- ``scenarios``: perfiles sintéticos multi-turno con fact-tracker.
- ``fixtures``: generador determinista de adjuntos PDF calibrados.
- ``metrics``: métricas binarias sobre ``turn_observed`` y el snapshot de sesión.
- ``run``: CLI que orquesta la campaña y vuelca el CSV.
"""

from evals.stress.scenarios import Scenario, Turn, all_scenarios, select_scenarios

__all__ = ["Scenario", "Turn", "all_scenarios", "select_scenarios"]
