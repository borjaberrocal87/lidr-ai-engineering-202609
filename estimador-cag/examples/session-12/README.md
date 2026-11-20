# Ejercicio Sesión 12 — Capa agéntica

Material para construir y depurar el agente de estimación (ver
[`../../specs/sesion-12-capa-agentica-bucle-manual.md`](../../specs/sesion-12-capa-agentica-bucle-manual.md)).

| Archivo | Origen | Papel |
|---|---|---|
| `sample_transcript_simple.txt` | Repo de referencia `session_12` | Un solo componente; depuración barata del bucle. |
| `sample_transcript_complex.txt` | Repo de referencia `session_12` | Cuatro componentes; input de los criterios de aceptación. |
| `calculate_estimate_skeleton.py` | Repo de referencia `session_12` | Esqueleto del modelo de costes determinista; se completa e integra. |
| `offline_budget_stub.py` | Propio | Backend de búsqueda enlatado para depurar el bucle sin Docker ni base de datos. |

`offline_budget_stub.py` **no** sustituye al pipeline de recuperación: el camino de
entrega es `search_budgets` → `retrieve()` sobre los corpus ingeridos. El stub solo
sirve para iterar rápido sobre la mecánica del bucle.

Ejecución de entrega (con el stack levantado y el corpus ingerido):

```bash
docker compose exec api python scripts/run_agent_s12.py \
  examples/session-12/sample_transcript_complex.txt \
  --model gpt-5 --effort medium \
  --out examples/session-12/trace_complex.txt
```
