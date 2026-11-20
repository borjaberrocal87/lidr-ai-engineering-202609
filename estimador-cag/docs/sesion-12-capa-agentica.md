# Sesión 12 — Capa agéntica: agente de estimación con bucle manual

Implementación de la spec
[`../specs/sesion-12-capa-agentica-bucle-manual.md`](../specs/sesion-12-capa-agentica-bucle-manual.md).

## Qué se construyó

| Pieza | Dónde |
|---|---|
| Contratos (argumentos, traza, informe) | `app/generation/agentic/estimation_agent/schemas.py` |
| Tools + schemas planos Responses + dispatch | `app/generation/agentic/estimation_agent/tools.py` |
| Bucle manual razona → actúa → observa | `app/generation/agentic/estimation_agent/runner.py` |
| Cliente Responses async | `get_async_openai_client()` en `app/dependencies.py` |
| Backend de `search_budgets` sobre `retrieve()` | `get_budget_search_backend()` en `app/dependencies.py` |
| Settings del agente | `agent_*` en `app/config.py` |
| Endpoint | `POST /api/v1/agent/estimate/from-transcript` (`app/api/agent_estimation.py`) |
| Script CLI | `scripts/run_agent_s12.py` |
| UI de trazas | `cliente-web/` → `/agent-estimations/new` |
| Material del ejercicio | `examples/session-12/` |
| Tests | `tests/test_agent_tools.py`, `tests/test_agent_runner.py`, `tests/test_agent_endpoint.py` |

Las tools que ve el modelo son tres: `search_budgets`, `calculate_estimate` y
`validate_estimate`. La calidad de sus descripciones vive en `tools.py`; son lo
único que el modelo lee para decidir cuándo usarlas.

## Cómo se ejecuta

Depuración barata del bucle (sin base de datos, backend enlatado):

```bash
cd estimador-cag
uv run python scripts/run_agent_s12.py \
  examples/session-12/sample_transcript_simple.txt \
  --model gpt-5-mini --effort minimal --stub
```

Ejecución con retrieval real (stack levantado y corpus ingerido):

```bash
docker compose exec api python scripts/run_agent_s12.py \
  examples/session-12/sample_transcript_simple.txt \
  --model gpt-5-mini --effort minimal
```

Ejecución de entrega:

```bash
docker compose exec api python scripts/run_agent_s12.py \
  examples/session-12/sample_transcript_complex.txt \
  --model gpt-5 --effort medium \
  --out examples/session-12/trace_complex.txt
```

Vía HTTP:

```bash
curl -s -X POST http://localhost:8000/api/v1/agent/estimate/from-transcript \
  -H 'Content-Type: application/json' \
  --data "$(jq -Rn --rawfile t examples/session-12/sample_transcript_complex.txt '{transcript:$t}')" | jq
```

La UI (`http://localhost:3000/agent-estimations/new`) ejecuta el agente, pinta la
traza paso a paso y descarga la ejecución como `.txt`.

## Traza de la ejecución de entrega

> **Pendiente de ejecución con credenciales.** La traza real se genera con el
> comando de entrega de arriba y queda en `examples/session-12/trace_complex.txt`.
> No se incluye aquí una traza inventada: el entregable es la salida de la
> ejecución real sobre `sample_transcript_complex.txt`.

Qué debe mostrar (formato mínimo exigido, por paso):

```text
STEP 1
  reasoning:   <what the agent decided and why>
  action:      search_budgets({"query": "...", "filters": ...})
  observation: <summary of returned items>
```

Y al final, el informe consolidado: componentes con horas y fuentes, total,
suposiciones y nivel de confianza, más el número de iteraciones y el motivo de
parada.

## Decisiones de adaptación

- **Layering**: `estimation_agent/` no importa `generation.rag` ni
  `app.dependencies`. El backend que envuelve `retrieve()` se construye en el
  composition root y se inyecta; `tests/test_layering.py` sigue verde sin
  cambios.
- **Excepción a `LLMWrapper`**: el bucle habla con la Responses API cruda porque
  el ejercicio consiste en conducir `function_call` / `function_call_output` a
  mano. Está documentado en el docstring del runner.
- **Unidades**: engineer-hours de punta a punta. El pipeline RAG de S9–S11 sigue
  en engineer-days; son caminos separados.
- **Corpus de búsqueda**: `budget_component` + `historical_task`, porque ambos
  llevan `estimated_hours` y los componentes de presupuesto son las referencias
  análogas que un componente de transcripción necesita.
- **`validate_estimate` obligatoria** como último paso, con guardrails de S4:
  componente sin referencia, horas fuera del rango implícito, total incoherente,
  total no positivo o desproporcionado.
- **Sin persistencia**: cada ejecución es un bucle fresco; no hay tabla ni
  migración.

## Criterios de aceptación

Con `sample_transcript_complex.txt`, `gpt-5` y effort `medium`:

- [ ] Identifica más de un componente y hace más de una llamada a `search_budgets`.
- [ ] Llama a `calculate_estimate` con los componentes y sus referencias.
- [ ] Llama a `validate_estimate` como último paso.
- [ ] Termina por sí solo (`stopped_reason="completed"`), sin agotar
      `max_iterations`.
- [ ] El informe es coherente: total = suma de componentes, sin `unbudgeted`
      silenciosos, suposiciones explícitas.
- [ ] La traza muestra razonamiento + acción + observación por paso, y la UI la
      pinta y la descarga como `.txt`.

Los criterios de mecánica (schema plano, `call_id`, parada, validación de
argumentos, no reimplementar retrieval) están cubiertos por
`tests/test_agent_*` y por `tests/test_layering.py`.

## Nota de comportamiento

_(Rellenar tras la ejecución real: cómo repartió las búsquedas por componente, cómo
reaccionó a cada observación y si paró solo a la primera.)_
