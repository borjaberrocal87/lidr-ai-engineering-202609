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

Ejecución real sobre `sample_transcript_complex.txt` con `gpt-5` y effort `medium`,
tal como la genera `scripts/run_agent_s12.py` (también disponible en
`examples/session-12/trace_complex.txt`):

```text
==============================================================================
AGENT TRACE
==============================================================================
  model=gpt-5   iterations=5   stopped=completed

STEP 1
  reasoning:   **Estimating effort for components**

I need to create an effort estimate following a structured method. First, I'll break down the components listed in the transcript, like the backend API, SAP ERP integration, mobile driver app, and analytics dashboard. For each of these, I should call search_budgets separately to get the hours associated with them. Then, I'll use calculate_estimate to determine the effort for each component while considering specialized filters like sectors and component types. Finally, I’ll validate the estimate based on the computed numbers. **Interpreting project components**

I need to interpret the tools mentioned and start by listing the components. First, there's the backend with API REST for managing orders, tracking shipments, and calculating tariffs, which needs to expose an API. Then, we have the SAP ERP integration for syncing billings and customer data. Next is the mobile app for delivery drivers, which will function offline and clearly present the day's routes. Lastly, I’ll outline the analytics panel with KPIs and alerts. I’ll keep my assumptions in check and avoid adding extra components. **Searching budgets per component**

I’m ready to proceed with searching budgets for each component. I’ll call functions.search_budgets using appropriate query strings in either Spanish or English. For the backend, I’ll use the phrase about managing orders and tracking shipments with a REST API. For the SAP integration, I'll mention billing and customer syncing with IDoc mapping. Next, I’ll describe the mobile app for drivers and its features, and finally outline the analytics dashboard with its KPIs and trends. Each search will include relevant filters.
  action:      search_budgets({"query": "Greenfield logistics business backend (TMS-like): order management, route assignment, shipment tracking, tariff calculation, exposing a REST API for other components.", "filters": {"sectors": ["logistics"], "component_type": "backend"}})
  observation: 8 historical items for 'Greenfield logistics business backend (TMS-like): order management, route assignment, shipment tracking, tariff calculation, exposing a REST API for other components.'; hours=[24, 17, 24, 31, 25, 23, 29, 33]

STEP 2
  reasoning:   (parallel tool call in the same turn as STEP 1)
  action:      search_budgets({"query": "SAP ERP integration for logistics: synchronize billing, customers, and item master; IDoc mapping; middleware layer; retry handling and resiliency.", "filters": {"sectors": ["logistics"], "component_type": "erp integration"}})
  observation: 8 historical items for 'SAP ERP integration for logistics: synchronize billing, customers, and item master; IDoc mapping; middleware layer; retry handling and resiliency.'; hours=[37, 52, 41, 25, 25, 33, 24, 17]

STEP 3
  reasoning:   (parallel tool call in the same turn as STEP 1)
  action:      search_budgets({"query": "Driver courier mobile app for logistics (Android and iOS): daily route sheet, delivery confirmation with signature and photo, offline mode with background sync; simple UX.", "filters": {"sectors": ["logistics"], "component_type": "mobile app"}})
  observation: 8 historical items for 'Driver courier mobile app for logistics (Android and iOS): daily route sheet, delivery confirmation with signature and photo, offline mode with background sync; simple UX.'; hours=[29, 33, 28, 24, 53, 12, 23, 20]

STEP 4
  reasoning:   (parallel tool call in the same turn as STEP 1)
  action:      search_budgets({"query": "Logistics analytics dashboard: KPIs on-time deliveries, cost per route, incidents; trend charts; filters by zone and customer; alerting when metrics deviate.", "filters": {"sectors": ["logistics"], "component_type": "analytics dashboard"}})
  observation: 8 historical items for 'Logistics analytics dashboard: KPIs on-time deliveries, cost per route, incidents; trend charts; filters by zone and customer; alerting when metrics deviate.'; hours=[47, 42, 36, 25, 44, 28, 34, 51]

STEP 5
  reasoning:   **Calculating estimates and medians**

I need to call `calculate_estimate` with the component names and reference hours to compute the medians. For the backend, I have the reference hours sorted, and the median is calculated from the 4th and 5th numbers, which gives me 24.5 after adding contingency. For SAP integration, I sort the references and find the median is 29.0. Lastly, for the mobile app, I sort again, and the median is 26. Time to let the tool handle the next steps! **Calculating analytics estimate**

I’m looking at the analytics numbers, which I need to sort first. Once sorted, I can find the median by averaging the 4th and 5th values, which gives me 39. After that, I should consider adding a contingency, potentially 20% or a fixed amount of hours. The tool will handle the calculations for these amounts. Now I just need to call `calculate_estimate` with the relevant arrays to get the final estimates!
  action:      calculate_estimate({"components": [{"name": "Backend de negocio con API (pedidos, rutas, seguimiento, tarifas)", "reference_amounts": [24, 17, 24, 31, 25, 23, 29, 33]}, {"name": "Integración con ERP SAP (IDocs, middleware, reintentos)", "reference_amounts": [37, 52, 41, 25, 25, 33, 24, 17]}, {"name": "App móvil repartidores (Android+iOS, firma/foto, offline)", "reference_amounts": [29, 33, 28, 24, 53, 12, 23, 20]}, {"name": "Panel de analítica (KPIs, filtros, alertas)", "reference_amounts": [47, 42, 36, 25, 44, 28, 34, 51]}]})
  observation: total=136.2h across 4 components

STEP 6
  reasoning:   **Calculating estimates with contingencies**

I noticed the tool uses a 15% contingency factor, which is interesting. The estimated hours seem to exceed the median values by specific amounts. For example, the backend median is 24.5, and after applying the contingency, it becomes 28.2. Similarly, for SAP, it’s 33.3, and for the app and analytics, the totals are 29.9 and 44.8, respectively. The grand total is 136.2. Now, I just need to run the validate_estimate as the final step.
  action:      validate_estimate({"components": [{"name": "Backend de negocio con API (pedidos, rutas, seguimiento, tarifas)", "estimated_hours": 28.2, "reference_amounts": [24, 17, 24, 31, 25, 23, 29, 33]}, {"name": "Integración con ERP SAP (IDocs, middleware, reintentos)", "estimated_hours": 33.3, "reference_amounts": [37, 52, 41, 25, 25, 33, 24, 17]}, {"name": "App móvil repartidores (Android+iOS, firma/foto, offline)", "estimated_hours": 29.9, "reference_amounts": [29, 33, 28, 24, 53, 12, 23, 20]}, {"name": "Panel de analítica (KPIs, filtros, alertas)", "estimated_hours": 44.8, "reference_amounts": [47, 42, 36, 25, 44, 28, 34, 51]}], "total_hours": 136.2})
  observation: estimate passed all guardrails

==============================================================================
FINAL REPORT  (iterations=5, stopped=completed)
==============================================================================
  - Backend de negocio con API (pedidos, rutas, seguimiento, tarifas): 28.2h  [sources: 642, 1658, 1380, 1657, 1642, 635, 1392, 1491]
      Mediana de 8 referencias de backend logístico similares (API + núcleo de negocio) con +15% de contingencia por greenfield.
  - Integración con ERP SAP (IDocs, middleware, reintentos): 33.3h  [sources: 1494, 1646, 650, 1671, 1642, 1491, 657, 659]
      Basada en 8 integraciones SAP con mapeo de IDocs y capa de reintentos, añadiendo +15% por riesgos habituales en datos maestros y resiliencia.
  - App móvil repartidores (Android+iOS, firma/foto, offline): 29.9h  [sources: 631, 1377, 1488, 1639, 632, 652, 635, 653]
      Derivada de 8 apps de reparto con confirmación y modo offline; contempla almacenamiento local y sincronización, con +15% por casos de conectividad.
  - Panel de analítica (KPIs, filtros, alertas): 44.8h  [sources: 1391, 1664, 1663, 1389, 1390, 1659, 1489, 1378]
      Tomada de 8 paneles logísticos con KPIs y alertas; incluye ETL ligero desde el backend y filtros, con +15% por definición de métricas y umbrales.

  TOTAL: 136.2h    confidence: medium
  assumptions:
    - Alcance MVP de cada componente centrado en las funciones descritas.
    - Stack tentativo: backend en Python; front web en React; app móvil híbrida (React Native/Flutter) aceptable; base de datos relacional (PostgreSQL).
    - SAP: uso de IDocs estándar (p. ej., DEBMAS, MATMAS, INVOIC); middleware con colas y reintentos; sin customizaciones SAP profundas ni BAPIs complejas fuera de alcance.
    - Asignación de rutas por reglas simples (zonas/capacidad/horarios); no incluye optimización avanzada tipo solver VRP.
    - App móvil: autenticación simple; caché local y sincronización con resolución básica de conflictos (last-write-wins); captura de firma y fotos en dispositivo; sin navegación giro a giro ni chat.
    - Analítica: KPIs básicos (entregas a tiempo, coste por ruta, incidencias), tendencias y filtros por zona/cliente; alertas por umbrales estáticos; sin data warehouse dedicado ni modelado dimensional complejo.
    - No incluido: hardening/seguridad avanzada, SSO corporativo, auditoría exhaustiva, CI/CD y observabilidad avanzadas, soporte multidioma, formación extensa, ni migración masiva de históricos.
    - QA: pruebas funcionales y UAT básicos; sin automatización E2E extensa.
    - Cumplimiento: RGPD estándar para datos e imágenes; se asume que no hay requisitos regulatorios adicionales.
```

Lectura rápida: el agente descompone el proyecto en sus cuatro componentes y lanza
las cuatro búsquedas **en paralelo en el primer turno** (un solo bloque de
razonamiento, tres de las calls marcadas como paralelas), consolida con
`calculate_estimate`, pasa `validate_estimate` y para sola. Total: 136,2 h.

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

- [x] Identifica más de un componente y hace más de una llamada a `search_budgets`.
- [x] Llama a `calculate_estimate` con los componentes y sus referencias.
- [x] Llama a `validate_estimate` como último paso.
- [x] Termina por sí solo (`stopped_reason="completed"`), sin agotar
      `max_iterations`.
- [x] El informe es coherente: total = suma de componentes, sin `unbudgeted`
      silenciosos, suposiciones explícitas.
- [x] La traza muestra razonamiento + acción + observación por paso, y la UI la
      pinta y la descarga como `.txt`.

Los criterios de mecánica (schema plano, `call_id`, parada, validación de
argumentos, no reimplementar retrieval) están cubiertos por
`tests/test_agent_*` y por `tests/test_layering.py`.

## Nota de comportamiento

El agente entendió la transcripción a la primera: identificó los cuatro componentes
sin añadir ni fusionar piezas y agrupó las cuatro búsquedas en un único turno de
razonamiento (STEP 1 con tres calls paralelas), con queries específicas por
componente y filtro de sector `logistics`. No hizo falta reintentar ninguna
búsqueda: `validate_estimate` pasó limpio en el primer intento. La única decisión
discutible es de negocio, no de mecánica: el agente costeó cada componente como un
MVP y dejó fuera explícitamente hardening, SSO, CI/CD y migración de históricos,
que es justo lo que el enunciado pide que se declare como suposición en lugar de
inventarlo. Paró sola en 5 iteraciones, muy por debajo del máximo de 10.
