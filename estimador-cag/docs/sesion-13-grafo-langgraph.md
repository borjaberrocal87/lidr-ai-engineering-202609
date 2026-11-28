# Sesión 13 — El flujo de estimación como grafo LangGraph

## Qué se construyó

El flujo de estimación de la Sesión 12 deja de ser un bucle manual y pasa a ser un
**grafo explícito de LangGraph** dentro del servicio IA, con estado tipado,
persistencia por `thread_id` y observabilidad por nodo.

- **Paquete** `app/generation/agentic/estimation_graph/`:
  - `state.py` — `EstimationState` (`TypedDict`, `total=False`) con dos reducers
    acumuladores (`budget_matches` y `errors`, `Annotated[..., operator.add]`) y un
    tercero propio de observabilidad (`node_runs`).
  - `schemas.py` — contratos Pydantic: salida estructurada de los nodos LLM,
    estimación determinista y el contrato HTTP.
  - `nodes.py` — los cinco nodos como **fábricas de clausuras** que reciben
    `LLMWrapper` y `BudgetSearchBackend` inyectados. Ninguno importa
    `app.dependencies` ni `app.generation.rag`: el layering se mantiene intacto.
  - `build.py` — cableado secuencial + arista condicional del Nivel 3
    (`validated` / `needs_review`, ambas a `END` de momento).
  - `checkpointer.py` — `AsyncPostgresSaver` sobre el Postgres de pgvector, con
    `saver_conninfo()` que pela el token de driver del `database_url`.
  - `observability.py` — Logfire tolerante (`send_to_logfire="if-token-present"`),
    un `span("node: …")` por nodo vía `traced_node`.
  - `render.py` — resumen de la ejecución (timeline → requisitos → componentes →
    matches → estimación → status), compartido por endpoint y CLI.
- **Endpoint** `POST /api/v1/graph/estimate/from-transcript` (`app/api/graph_estimation.py`),
  con `thread_id = estimation_id` (uuid4 si no viene), 503 sin grafo, 502 en fallo y
  `run_text` renderizado en servidor. Los endpoints existentes no cambian.
- **Lifespan** (`app/main.py`): configura Logfire, abre el checkpointer, hace
  `setup()` idempotente y compila el grafo una vez en `app.state.graph`. Si el
  Postgres del checkpointer no está, degrada a 503 en el endpoint del grafo sin
  tumbar el resto del servicio.
- **Script** `scripts/run_graph_s13.py` con `--memory`, `--stub`, `--transcript`,
  `--estimation-id` y `--out`.
- **UI** en `cliente-web/`: `graphEstimationClient.js`, `routes/graphEstimations.js`,
  `views/graph-estimations/` con timeline de nodos (`_pipeline.ejs`), estimación y
  avisos (`_estimate.ejs`) y descarga del `run_text` en `.txt`.
- **Tests**: `tests/test_graph_{state,nodes,build,endpoint}.py` (18 casos, con
  dobles, sin red ni Postgres) y `cliente-web/test/graphEstimations.test.js`
  (4 casos con supertest).

**Qué se reutilizó de S9–S12** (sin reimplementar):

- El backend de recuperación de S12 (`make_budget_search_backend`) — híbrido +
  reranking de S9–S10 sobre la colección `budget`.
- El modelo de costes determinista de S12 (`estimation_agent/tools.py`): mediana por
  componente + 15 % de contingencia, `unbudgeted` a 0 h.
- Los guardrails de S12 adaptados a función pura sobre el estado.
- `LLMWrapper.complete_structured` para los dos nodos LLM (Instructor + LiteLLM).
- Las transcripciones de `examples/session-12/` y el stub offline de recuperación.

**Decisiones propias** (frente al repo de referencia): grafo en `agentic/` con
dependencias inyectadas (la referencia auto-cablea `app.dependencies` desde los
nodos); **engineer-hours** de punta a punta (sin conversión ÷8 a días); y
`generate_estimate` **determinista**, así que una ejecución hace solo **dos
llamadas al LLM** por ejecución, ambas con `gpt-5-mini`.

## Ejecución de entrega (real)

Comando (desde `estimador-cag/`, con el stack levantado y ambos corpus ingeridos):

```bash
uv run python scripts/run_graph_s13.py \
    --transcript examples/session-12/sample_transcript_complex.txt \
    --out examples/session-13/trace_complex.txt
```

Resultado: **7 componentes**, **53 referencias históricas** acumuladas por el nodo
`search_budgets` (secuencial) y **267,1 h** de estimación total con
`status="validated"`. Traza completa (idéntica a
[`examples/session-13/trace_complex.txt`](../examples/session-13/trace_complex.txt)):

```text
==============================================================================
SESSION 13 — ESTIMATION GRAPH RUN
==============================================================================
estimation_id : s13-sample_transcript_complex
status        : validated

NODE TIMELINE
  - extract_requirements: 18553.958 ms
  - classify_components: 16245.023 ms
  - search_budgets: 4595.61 ms
  - generate_estimate: 0.543 ms
  - validate_and_consolidate: 0.184 ms

REQUIREMENTS
  - Build a unified logistics platform named RUTA that consolidates functionality currently spread across four separate sites.
  - Implement a backend system for order management.
  - Implement route assignment and scheduling functionality in the backend.
  - Implement shipment tracking (real-time status tracking) in the backend.
  - Implement a tariff and rate calculation engine in the backend.
  - Expose a programmatic API for all backend functionality to be consumed by other components.
  - Integrate with SAP ERP to synchronize billing/invoicing data.
  - Integrate with SAP ERP to synchronize customer master data.
  - Integrate with SAP ERP to synchronize product/item master data.
  - Implement IDoc mapping and processing for SAP message exchange.
  - Provide a middleware layer/component to manage SAP integration workflows.
  - Implement retry, error handling and reconciliation mechanisms for SAP integration.
  - Build a mobile application for delivery drivers targeting both Android and iOS.
  - In the mobile app, display each driver's daily route/itinerary.
  - In the mobile app, allow delivery confirmation including electronic signature capture.
  - In the mobile app, allow attaching photos as proof-of-delivery.
  - Support offline mode in the mobile app with local data storage and synchronization when connectivity is restored.
  - Design the mobile app user interface for non-technical users with a very simple, easy-to-use workflow.
  - Implement a management analytics/dashboard module with interactive dashboards.
  - Include on-time delivery rate as a KPI in the analytics dashboards.
  - Include cost-per-route reporting and KPI in the analytics dashboards.
  - Include incident tracking and reporting in the analytics dashboards.
  - Provide trend visualizations (time-series) in the analytics dashboards.
  - Enable dashboard filtering by geographic zone and by customer.
  - Implement alerting/notifications when KPIs or other metrics deviate from defined thresholds.
  - Deliver the backend and SAP integration in the initial project phase and schedule the mobile app and analytics in subsequent phases.
  - Use technologies compatible with the internal team's skills (prefer backend in Python and frontend in React) unless a better alternative is justified.
  - Propose and justify a database solution as part of the technical design.
  - Provide an effort estimation broken down by component and a total estimate for the entire project.

COMPONENTS
  - RUTA Unified Web Platform [frontend]
  - Backend Services (Orders, Routing, Tracking, Tariff, API) [backend]
  - SAP Integration & Middleware [integration]
  - Mobile Driver App [mobile]
  - Analytics & Dashboards [analytics]
  - Database & Data Platform [infrastructure]
  - Infrastructure, DevOps & Project Phasing [infrastructure]

BUDGET MATCHES (accumulator field)
  - RUTA Unified Web Platform: 36.0h (ref TASK-2025-0006, distance 0.4877)
  - RUTA Unified Web Platform: 23.0h (ref TASK-2025-0006, distance 0.4885)
  - RUTA Unified Web Platform: 16.0h (ref TASK-2023-0005, distance 0.4889)
  - RUTA Unified Web Platform: 23.0h (ref TASK-2023-0005, distance 0.4896)
  - RUTA Unified Web Platform: 18.0h (ref TASK-2022-0027, distance 0.4934)
  - RUTA Unified Web Platform: 40.0h (ref TASK-2022-0027, distance 0.4938)
  - RUTA Unified Web Platform: 13.0h (ref TASK-2025-0003, distance 0.4949)
  - RUTA Unified Web Platform: 25.0h (ref TASK-2025-0003, distance 0.4959)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 36.0h (ref TASK-2024-0038, distance 0.5106)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 34.0h (ref TASK-2024-0054, distance 0.5137)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 47.0h (ref TASK-2024-0003, distance 0.5171)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 46.0h (ref TASK-2022-0032, distance 0.5188)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 40.0h (ref TASK-2024-0012, distance 0.519)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 31.0h (ref TASK-2023-0057, distance 0.5205)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 34.0h (ref TASK-2025-0017, distance 0.522)
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 38.0h (ref TASK-2022-0001, distance 0.523)
  - SAP Integration & Middleware: 39.0h (ref TASK-2022-0033, distance 0.5884)
  - SAP Integration & Middleware: 52.0h (ref TASK-2023-0014, distance 0.5926)
  - SAP Integration & Middleware: 45.0h (ref TASK-2025-0041, distance 0.5931)
  - SAP Integration & Middleware: 41.0h (ref TASK-2022-0007, distance 0.5942)
  - SAP Integration & Middleware: 31.0h (ref TASK-2023-0057, distance 0.5953)
  - SAP Integration & Middleware: 29.0h (ref TASK-2024-0003, distance 0.5963)
  - SAP Integration & Middleware: 37.0h (ref TASK-2023-0050, distance 0.5967)
  - SAP Integration & Middleware: 24.0h (ref TASK-2023-0046, distance 0.5968)
  - Mobile Driver App: 29.0h (ref TASK-2024-0018, distance 0.4434)
  - Mobile Driver App: 33.0h (ref TASK-2025-0045, distance 0.4547)
  - Mobile Driver App: 28.0h (ref TASK-2023-0050, distance 0.4641)
  - Mobile Driver App: 24.0h (ref TASK-2023-0057, distance 0.4686)
  - Mobile Driver App: 180.0h (ref BUD-2024-007, distance 0.583)
  - Analytics & Dashboards: 36.0h (ref TASK-2022-0034, distance 0.3853)
  - Analytics & Dashboards: 34.0h (ref TASK-2023-0046, distance 0.389)
  - Analytics & Dashboards: 45.0h (ref TASK-2023-0029, distance 0.3898)
  - Analytics & Dashboards: 27.0h (ref TASK-2025-0017, distance 0.3925)
  - Analytics & Dashboards: 27.0h (ref TASK-2023-0044, distance 0.3927)
  - Analytics & Dashboards: 39.0h (ref TASK-2022-0032, distance 0.3976)
  - Analytics & Dashboards: 44.0h (ref TASK-2023-0020, distance 0.3988)
  - Analytics & Dashboards: 41.0h (ref TASK-2023-0024, distance 0.3998)
  - Database & Data Platform: 35.0h (ref TASK-2023-0007, distance 0.5052)
  - Database & Data Platform: 32.0h (ref TASK-2025-0002, distance 0.5154)
  - Database & Data Platform: 18.0h (ref TASK-2025-0042, distance 0.5177)
  - Database & Data Platform: 53.0h (ref TASK-2022-0027, distance 0.5202)
  - Database & Data Platform: 27.0h (ref TASK-2024-0016, distance 0.5248)
  - Database & Data Platform: 35.0h (ref TASK-2023-0022, distance 0.5263)
  - Database & Data Platform: 43.0h (ref TASK-2025-0042, distance 0.5275)
  - Database & Data Platform: 36.0h (ref TASK-2022-0030, distance 0.5276)
  - Infrastructure, DevOps & Project Phasing: 40.0h (ref TASK-2024-0038, distance 0.4302)
  - Infrastructure, DevOps & Project Phasing: 29.0h (ref TASK-2022-0007, distance 0.4339)
  - Infrastructure, DevOps & Project Phasing: 32.0h (ref TASK-2023-0005, distance 0.437)
  - Infrastructure, DevOps & Project Phasing: 34.0h (ref TASK-2025-0042, distance 0.437)
  - Infrastructure, DevOps & Project Phasing: 34.0h (ref TASK-2023-0057, distance 0.4373)
  - Infrastructure, DevOps & Project Phasing: 22.0h (ref TASK-2024-0054, distance 0.4373)
  - Infrastructure, DevOps & Project Phasing: 23.0h (ref TASK-2024-0023, distance 0.4384)
  - Infrastructure, DevOps & Project Phasing: 38.0h (ref TASK-2024-0023, distance 0.4393)
  (53 matches total)

ESTIMATE
  - RUTA Unified Web Platform: 26.4h — Median of 8 historical reference(s) plus a 15% contingency.
  - Backend Services (Orders, Routing, Tracking, Tariff, API): 42.5h — Median of 8 historical reference(s) plus a 15% contingency.
  - SAP Integration & Middleware: 43.7h — Median of 8 historical reference(s) plus a 15% contingency.
  - Mobile Driver App: 33.3h — Median of 5 historical reference(s) plus a 15% contingency.
  - Analytics & Dashboards: 43.1h — Median of 8 historical reference(s) plus a 15% contingency.
  - Database & Data Platform: 40.2h — Median of 8 historical reference(s) plus a 15% contingency.
  - Infrastructure, DevOps & Project Phasing: 37.9h — Median of 8 historical reference(s) plus a 15% contingency.
  TOTAL: 267.1h
```

### Lectura de los tiempos

El coste está en los dos nodos LLM: `extract_requirements` (18,6 s) y
`classify_components` (16,2 s). `search_budgets` tarda 4,6 s **en secuencial** para
7 componentes (embedding + recuperación por componente); `generate_estimate` y
`validate_and_consolidate` son deterministas y no llegan al milisegundo. Ese 4,6 s
secuencial es justo el punto de partida del directo: paralelizarlo con la Send API
y medir el ahorro.

## Persistencia (checkpointer)

Misma base de datos del proyecto (`estimator` en el contenedor
`estimador-cag-estimator-postgres`); el checkpointer creó sus propias tablas, sin
migración de Alembic:

```text
 public | checkpoint_blobs      | table | estimator
 public | checkpoint_migrations | table | estimator
 public | checkpoint_writes     | table | estimator
 public | checkpoints           | table | estimator
```

La ejecución de entrega quedó persistida bajo su `thread_id`:

```text
           thread_id           | checkpoints
-------------------------------+-------------
 s13-sample_transcript_complex |           7
```

(7 checkpoints = uno por nodo más el estado inicial.)

## Observabilidad

Logfire está configurado con `send_to_logfire="if-token-present"`: cada nodo corre
dentro de un `span("node: <name>")`, y FastAPI + httpx quedan instrumentados en el
servicio. **En esta ejecución no había `LOGFIRE_TOKEN` configurado**, así que los
spans corrieron en local y no se exportaron; la timeline visible (`node_runs` con
nombre y duración por nodo) es la que aparece arriba y la que pinta la UI. Para
obtener la traza exportada con un span por nodo basta con poner el token y repetir
la ejecución:

```bash
LOGFIRE_TOKEN=... uv run python scripts/run_graph_s13.py \
    --transcript examples/session-12/sample_transcript_complex.txt
```

## Nota de comportamiento

Con la transcripción compleja el grafo descompone el proyecto en 7 componentes de
naturaleza distinta (frontend, backend, integración SAP, móvil, analítica, datos e
infraestructura) y recupera referencias para cada uno por separado, acumulando 53
matches en `budget_matches`. La consolidación determinista usa la mediana de cada
componente más el 15 % de contingencia y suma 267,1 h; la validación no encontró
ningún componente `unbudgeted` ni desviaciones de rango, así que el `status` salió
`validated` a la primera.

## Extra — Marcador (web + app de resultados de fútbol)

Además de la transcripción de entrega, el repo incluye una segunda transcripción
compleja de otro dominio: `examples/session-13/sample_transcript_marcador.txt`
(plataforma de resultados de fútbol en vivo, con web y app móvil). Su ejecución
real está en `examples/session-13/trace_marcador.txt` (`thread_id=s13-marcador`):

- 13 componentes (ingesta de datos deportivos, histórico, backend, match center y
  APIs, tiempo real, notificaciones, cuentas, pagos, backoffice, web, apps
  móviles, analítica/observabilidad e infraestructura).
- 96 referencias históricas acumuladas por `search_budgets` (secuencial).
- `total_hours = 474,3` y `status = "needs_review"`: el componente *Match Center &
  Public/Internal APIs* quedó `unbudgeted` (el corpus no tenía una referencia
  suficientemente cercana), así que la validación no inventó un número y enrutó
  por la rama `needs_review` del Nivel 3. Es el caso que el directo resolverá con
  fallback/retry y, llegado el caso, intervención humana.
- Persistida en el checkpointer con 7 checkpoints (uno por nodo más el inicial).

Reproducción:

```bash
uv run python scripts/run_graph_s13.py \
    --transcript examples/session-13/sample_transcript_marcador.txt \
    --estimation-id s13-marcador \
    --out examples/session-13/trace_marcador.txt
```
