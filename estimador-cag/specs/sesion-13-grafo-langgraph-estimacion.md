# Spec — Proyecto 7: El flujo de estimación como grafo LangGraph: estado, persistencia y observabilidad (Sesión 13)

## Objetivo

En la Sesión 12 construiste a mano un bucle agéntico sobre el pipeline RAG: un LLM decidía entre
`search_budgets`, `calculate_estimate` y `validate_estimate`, ejecutaba tools y exponía su traza.
Funciona, pero el bucle manual se vuelve incómodo en cuanto el flujo tiene varios pasos, ramas
condicionales o necesita volver atrás: el control de flujo vive disperso en un `while` y en las
decisiones del modelo, y el estado de la ejecución se pierde al terminar la petición.

Esta sesión reexpresa ese flujo como un **grafo explícito de LangGraph** dentro del servicio IA.
Cinco nodos secuenciales, un estado compartido tipado con reducers acumuladores, persistencia por
`thread_id` sobre el Postgres del proyecto y una traza completa de observabilidad por ejecución.

Al terminar tendrás:

- Un paquete `app/generation/agentic/estimation_graph/` con `state.py` (el `TypedDict` del estado,
  con dos campos acumuladores vía `Annotated[..., operator.add]`), `schemas.py` (modelos Pydantic de
  entrada/salida de los nodos), `nodes.py` (los cinco nodos como funciones que reciben el estado y
  devuelven actualizaciones parciales), `build.py` (el cableado y el `compile`) y `checkpointer.py`
  (el `AsyncPostgresSaver` sobre el Postgres de pgvector).
- Un grafo `START → extract_requirements → classify_components → search_budgets →
  generate_estimate → validate_and_consolidate → END`, de momento **secuencial**, con el opcional
  del Nivel 3 (arista condicional de salida según el `status` de validación).
- El endpoint nuevo `POST /api/v1/graph/estimate/from-transcript`: misma forma de puertas afuera
  que el resto de estimadores — transcripción dentro, estimación estructurada con su `status`
  fuera. El backend de negocio ni se entera de que debajo hay un grafo.
- Persistencia real: cada ejecución lleva su `thread_id` (el identificador de la estimación) y el
  checkpointer persiste el estado del grafo en las tablas que él mismo crea en el Postgres del
  proyecto. Sin infraestructura nueva, sin migraciones de Alembic.
- Observabilidad con **Logfire**: configuración tolerante a falta de token, instrumentación de
  FastAPI y httpx (que captura las llamadas al LLM), y **un span por nodo**. Además, un acumulador
  propio `node_runs` con nombre y duración de cada nodo, para que la timeline sea visible sin
  depender de que Logfire esté configurado.
- Un script CLI (`scripts/run_graph_s13.py`) con modos `--memory` y `--stub`, una UI en
  `cliente-web/` con la timeline de nodos, y `docs/sesion-13-grafo-langgraph.md` con la traza real
  de la ejecución de entrega sobre `sample_transcript_complex.txt`.

El directo parte de aquí: paralelizar `search_budgets` con la Send API, robustez (reintentos,
fallback, timeouts) e intervención humana en la validación. Por eso la pre-sesión **no** toca nada
de eso (ver "Lo que NO toca todavía").

> **Alcance estricto.** Esta sesión no modifica el bucle agéntico de S12 (sigue vivo y servido por
> su endpoint), ni el pipeline RAG, ni el wizard, ni los corpus. El grafo es un camino **nuevo**
> que reutiliza las piezas de S9–S12 como funciones dentro de nodos. Tampoco hay ejecución
> paralela, ni reintentos, ni HITL: secuencial, cableado y trazable.

## Reconciliación con el repo

El punto de partida es `main` con la Sesión 12 cerrada: agente de estimación con tools, traza
visible en la UI y endpoint `/api/v1/agent/estimate/from-transcript`. El repo de referencia
(`LIDR-academy-temp/ai-engineering`, rama `session_13`) sirve de **guía de alcance**, no de
plantilla: de él se toma la forma del estado, la idea del checkpointer sobre el Postgres existente
y la pauta de "un span por nodo". Su estructura (`app/domain/graph/` con nodos auto-cableados), sus
unidades y su UI no se copian. Todo lo que sigue se diseña sobre los contratos y convenciones de
este repo.

**Decisiones estructurales propias:**

- **El grafo vive en `app/generation/agentic/estimation_graph/`**, hermano de `estimation_agent/`,
  `boss.py` y `critic.py`. La referencia lo pone en `app/domain/graph/`, pero aquí `domain` es la
  capa de contratos y conductores, y además la referencia incumple algo que aquí está prohibido:
  sus nodos importan `app.dependencies` directamente. `tests/test_layering.py` solo permite ese
  import en el composition root y los puentes; el grafo lo respeta sin pedir excepciones nuevas:
  **los nodos son fábricas de clausuras** (`make_search_budgets_node(backend)`, ...) que reciben
  `LLMWrapper` y `BudgetSearchBackend` inyectados, y el wiring vive en `app/dependencies.py` y en
  el lifespan de `app/main.py`. Mismo patrón que `runner.py` en S12.
- **Unidades: engineer-hours.** El corpus registra `estimated_hours`; S12 fijó horas para el agente
  y esta sesión mantiene esa decisión: el grafo trabaja en horas de punta a punta, sin la
  conversión ÷8 a engineer-days que hace la referencia. El pipeline RAG de S9–S11 sigue en días:
  caminos distintos, no se mezclan.
- **`generate_estimate` es determinista.** La referencia hace una tercera llamada al LLM para
  consolidar. Aquí no: el nodo reutiliza el modelo de costes de S12 (mediana de las referencias por
  componente × buffer de contingencia, componente sin referencias → `unbudgeted` con 0 h y total
  que no lo cuenta) importando las funciones puras de `estimation_agent/tools.py`. El grafo hace
  **dos** llamadas al LLM en total (extracción y clasificación), ambas por `LLMWrapper`: la
  excepción documentada de S12 (Responses API directa) era el ejercicio del bucle manual, y el
  bucle manual ya no existe.
- **`extract_requirements` y `classify_components` usan `LLMWrapper.complete_structured`**
  (LiteLLM + Instructor), el mismo camino de salida estructurada que usa el resto del repo. Con
  `model_override=settings.graph_extraction_model` (por defecto `gpt-5-mini`): son tareas pequeñas
  y el grafo debe ser barato de depurar.
- **Checkpointer asíncrono sobre el Postgres del proyecto.** `AsyncPostgresSaver` (psycopg3 async)
  creado desde `database_url` pelando el token de driver (`postgresql+psycopg://` →
  `postgresql://`), igual que `_async_database_url` hace el swap para SQLAlchemy. Sus tablas
  (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`) conviven con las de embeddings **sin
  migración de Alembic**: las crea `await checkpointer.setup()`, idempotente, una vez por arranque
  en el lifespan.
- **Configuración en `app/config.py`** con nuestro snake_case: `graph_extraction_model`,
  `graph_search_top_k`, `graph_search_distance_threshold`, `logfire_service_name`,
  `logfire_token`. Los dos primeros reutilizan los valores del agente como defecto (8 / 0.6).
- **Observabilidad tolerante.** `configure_logfire()` con `send_to_logfire="if-token-present"`: sin
  `LOGFIRE_TOKEN` los spans se ejecutan pero no se exportan, así que el servicio nunca depende de
  Logfire. Y como la entrega pide "una traza completa", el estado acumula `node_runs`
  (nombre + duración de cada nodo), que alimenta la timeline de la UI y el `.txt` de la ejecución
  sin depender de un token externo. Si prefieres LangSmith, el hueco está en
  `estimation_graph/checkpointer.py`/`observability` — pero se documenta cuál se usó; aquí se
  decide Logfire.
- **El grafo se compila una vez por arranque**, en el lifespan de `app/main.py`, y se guarda en
  `app.state.graph`. El router lo lee de ahí; si el arranque no pudo construirlo (p. ej. Postgres
  del checkpointer caído), el endpoint responde 503. No se compila por petición.
- **Contrato HTTP nuevo, existentes intactos.** `POST /api/v1/graph/estimate/from-transcript` es un
  camino adicional; `/estimate/*`, `/estimate/stages/*` y `/agent/estimate/*` siguen igual. La
  respuesta lleva `estimation_id`, `status`, `estimate`, `requirements`, `components`,
  `budget_matches`, `node_runs`, `errors` y un `run_text` renderizado en servidor (el patrón
  `trace_text` de S12: la UI pinta y descarga sin duplicar formato en JavaScript).
- **Material de la sesión en `examples/session-13/`** y transcripciones reutilizadas de
  `examples/session-12/` (la simple para depurar barato, la compleja para la entrega). El script es
  `scripts/run_graph_s13.py` (convención `_sNN`) y los tests son planos
  `tests/test_graph_*.py`, con dobles, sin red ni Postgres.

**Material tomado del repo de referencia:** nada de código. Solo la forma del estado tipado y la
pauta de instrumentación (un `logfire.span("node: ...")` por nodo), que el enunciado ya trae. Las
transcripciones ya estaban en el repo desde S12.

**Lo que se escribe propio** (y por qué no vale copiarlo de la referencia): todo — estado, schemas,
nodos con inyección, cableado, checkpointer con nuestro DSN, observabilidad, endpoint, script, UI,
tests y el documento con **nuestra** traza real.

### Piezas que se reutilizan

- `app/generation/agentic/estimation_agent/tools.py` — el modelo de costes determinista
  (mediana + contingencia, `unbudgeted` a 0 h) y la *idea* de los guardrails de
  `validate_estimate`: `generate_estimate` y `validate_and_consolidate` los importan/adaptan, no
  los reimplementan. Mismo paquete `agentic`: el import entre hermanos internos está permitido.
- `app/dependencies.py` — `make_budget_search_backend(top_k=..., distance_threshold=...)` de S12:
  el único punto donde el grafo toca el retrieval de S9–S10 (híbrido + reranking sobre la colección
  `budget`, `chunk_types=["budget_component", "historical_task"]`). Y `get_llm_wrapper()` para los
  nodos con LLM.
- `app/foundation/llm/` — `LLMWrapper.complete_structured` (salida estructurada con Instructor).
- `app/config.py` — `database_url` (fuente del DSN del checkpointer) y el patrón de settings
  snake_case.
- `app/generation/rag/stage_logging.py` — `current_request_id`: los logs del grafo quedan
  correlacionados con la petición HTTP igual que los del pipeline RAG.
- `examples/session-12/offline_budget_stub.py` — el stub offline de recuperación: el modo `--stub`
  del script lo reutiliza para depurar sin Docker/Postgres.
- `cliente-web/src/agentEstimationClient.js` + `routes/agentEstimations.js` +
  `views/agent-estimations/` — patrón exacto que replica la UI del grafo (cliente HTTP, rutas,
  vistas con parciales, descarga de la ejecución en `.txt`, test con supertest).
- `tests/test_agent_runner.py` — estilo de tests con dobles guionizados, sin red.

## Requisitos para el ejercicio

- Proyecto de la Sesión 12 en verde:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy app frontend
  ```

- Dependencias nuevas (gestor `uv`, sobre el `pyproject.toml` del servicio):

  ```bash
  uv add langgraph langgraph-checkpoint-postgres logfire
  ```

  Si `mypy` se queja de los stubs de `langgraph`, añade el override
  `[[tool.mypy.overrides]] module = "langgraph.*" ignore_missing_imports = true` — documentado en
  el commit, no silenciado a ciegas.

- Stack levantado desde `estimador-cag/`:

  ```bash
  docker compose up -d --build
  curl -s http://localhost:8000/health | jq
  ```

- Corpus ingerido de forma idempotente (el grafo busca sobre `budget_component` y
  `historical_task`):

  ```bash
  docker compose run --rm api python scripts/query_examples.py
  docker compose run --rm api python scripts/build_task_corpus.py --ingest
  ```

- Clave de OpenAI en `.env` (`OPEN_AI_KEY`): sin clave no hay extracción ni clasificación; los
  nodos LLM degradan igual que el resto del repo.

- Opcional pero recomendado para la entrega: `LOGFIRE_TOKEN` en `.env`. Sin él todo funciona y la
  timeline vive en `node_runs`; con él, además, tienes la traza exportada con un span por nodo.

## Lo que entra y lo que no entra

**Entra:**

- `app/generation/agentic/estimation_graph/` con `state.py`, `schemas.py`, `nodes.py`, `build.py` y
  `checkpointer.py` (+ `observability.py` si separas la configuración de Logfire; queda a tu
  criterio, pero debe existir un único punto de configuración).
- El estado tipado con `total=False` y **dos** campos acumuladores (`budget_matches` y `errors`,
  ambos `Annotated[list[...], operator.add]`) más el acumulador de observabilidad `node_runs`.
- Los cinco nodos como funciones puras `state → partial update`, envueltas por un helper de
  trazado que abre el span de Logfire, mide la duración y añade el registro a `node_runs`.
- `build_graph(...)` con cableado directo, `compile(checkpointer=...)` y (Nivel 3, opcional) la
  arista condicional `route_on_status`.
- `saver_conninfo()` + `open_checkpointer()` (context manager async) sobre el Postgres del
  proyecto; `await checkpointer.setup()` idempotente en el arranque.
- Configuración de Logfire tolerante (`if-token-present`) + `instrument_fastapi` +
  `instrument_httpx`.
- `get_estimation_graph_dependencies()` (o equivalente) en `app/dependencies.py` y la compilación
  del grafo en el lifespan de `app/main.py` (`app.state.graph`).
- `POST /api/v1/graph/estimate/from-transcript` en `app/api/graph_estimation.py`, con `thread_id =
  estimation_id` (uuid4 si no viene) y respuesta con `run_text` renderizado en servidor.
- Script CLI `scripts/run_graph_s13.py` con `--transcript`, `--estimation-id`, `--memory`,
  `--stub` y `--out`.
- UI en `cliente-web/`: `src/graphEstimationClient.js`, `src/routes/graphEstimations.js`
  (`GET /graph-estimations/new`, `POST /graph-estimations`), `views/graph-estimations/` con
  `new.ejs`, `show.ejs`, `_pipeline.ejs` (timeline de `node_runs`) y `_estimate.ejs`, enlace en el
  header y botón de descarga de `run_text` en `.txt`.
- Tests planos: `tests/test_graph_state.py`, `tests/test_graph_nodes.py`,
  `tests/test_graph_build.py`, `tests/test_graph_endpoint.py` (+ `test/graphEstimations.test.js`
  en `cliente-web`).
- `docs/sesion-13-grafo-langgraph.md` con la ejecución real sobre
  `sample_transcript_complex.txt` y `examples/session-13/trace_complex.txt` con la misma traza.

**No entra:**

- **Paralelismo**: `search_budgets` recorre los componentes uno tras otro. La Send API, medir el
  coste de lo secuencial y paralelizar es el caso estrella del directo.
- **Manejo de errores avanzado**: reintentos con backoff, nodo de fallback, timeouts por nodo,
  circuit breakers. Un fallo de recuperación de un componente es *soft* (apunta a `errors` y
  sigue), nada más.
- **HITL**: nada de `interrupt()` en la validación.
- **Optimización a partir de la traza**: la lectura fina para decidir qué tocar se hace juntos en
  el directo.
- Cambios en el bucle agéntico de S12, en el pipeline RAG, en el wizard, en los corpus o en los
  endpoints existentes.
- Migraciones de Alembic: el checkpointer gestiona sus propias tablas con `setup()`.
- Infraestructura nueva: ni otra base de datos ni otro servicio de Docker.

## Datos para el ejercicio

### Transcripciones

`examples/session-12/sample_transcript_simple.txt` y `sample_transcript_complex.txt` (ya en el
repo desde S12). La simple depura la mecánica del grafo gastando lo mínimo; la compleja —
multicomponente, el caso del enunciado — es el input de los criterios de aceptación y de la traza
de entrega.

### Modelo de costes

El de S12, sin rediseñar: por componente, **mediana** de las horas de referencia × **buffer de
contingencia** fijo y transparente; componente sin referencias → `unbudgeted` con 0 h (no se
inventa un número; `validate_and_consolidate` lo flaggeará); total = suma de componentes
presupuestados, redondeado a una decimal. Vive como funciones puras en
`estimation_agent/tools.py` y `generate_estimate` las invoca.

### El estado del grafo

`state.py` — la forma del enunciado, con nuestras dos adiciones (`estimation_id` para correlar
logs y `node_runs` para la timeline):

```python
from __future__ import annotations

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict


class Component(TypedDict):
    name: str
    category: str


class BudgetMatch(TypedDict):
    component: str
    reference_budget_id: Optional[str]
    amount: float  # engineer-hours of the historical reference
    distance: float


class NodeRun(TypedDict):
    name: str
    duration_ms: float


class EstimationState(TypedDict, total=False):
    transcript: str
    estimation_id: str
    requirements: list[str]
    components: list[Component]
    # Accumulators: a node returns only the NEW items; the reducer appends.
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    node_runs: Annotated[list[NodeRun], operator.add]
    estimate: Optional[dict]
    status: Optional[str]  # "validated" | "needs_review"
    errors: Annotated[list[str], operator.add]
```

`total=False` es lo que permite que cada nodo devuelva un dict parcial sin todas las claves y que
el `ainvoke` inicial lleve solo `transcript` (+ `estimation_id`).

## Ejercicio paso a paso

### Paso 0 — Dependencias y settings

1. `uv add langgraph langgraph-checkpoint-postgres logfire` y el override de mypy si hace falta.

2. Añade a `app/config.py`:

   ```python
   graph_extraction_model: str = "gpt-5-mini"
   graph_search_top_k: int = 8
   graph_search_distance_threshold: float = 0.6
   logfire_service_name: str = "estimador-cag"
   logfire_token: SecretStr | None = None
   ```

### Paso 1 — Estado

`estimation_graph/state.py` como el listado de arriba. La prueba de que los reducers funcionan
viene en los tests del Paso 9: un nodo que devuelve `{"budget_matches": [m1]}` tras otro que
devolvió `{"budget_matches": [m0]}` deja en el estado `[m0, m1]`, no `[m1]`.

### Paso 2 — Schemas

`estimation_graph/schemas.py`, modelos Pydantic con descripciones en inglés:

1. **Salida estructurada de los nodos LLM.** `RequirementsExtraction` (`requirements:
   list[str]`), `ComponentClassification` (`components: list[ComponentOut]` con `name` y
   `category`). Son lo que `complete_structured` valida con Instructor.

2. **Estimación.** `EstimateComponent` (`name`, `estimated_hours: float | None`, `unbudgeted:
   bool`, `reference_budget_ids: list[str]`, `rationale`) y `EstimateReport` (`components`,
   `total_hours: float`). En horas, sin conversiones. `generate_estimate` lo construye de forma
   determinista y lo serializa con `model_dump()` al campo `estimate` del estado.

3. **Contrato HTTP.** `GraphEstimateRequest` (`transcript: str`, `estimation_id: str | None`) y
   `GraphEstimateResponse` (`estimation_id`, `status`, `estimate`, `requirements`, `components`,
   `budget_matches`, `node_runs`, `errors`, `run_text`). Mismo estilo de validación y errores que
   el request del agente S12 (422 por FastAPI).

### Paso 3 — Nodos

`estimation_graph/nodes.py`. Cada nodo es una **fábrica de clausura** que recibe sus dependencias
inyectadas y devuelve la función nodo. Todos comparten la misma forma: reciben el estado,
devuelven una actualización parcial. Y todos pasan por el helper de trazado:

```python
def traced_node(name: str, fn: NodeFn) -> NodeFn:
    """Wrap a node: one logfire span per node + a NodeRun appended to the state."""
    @functools.wraps(fn)
    async def wrapper(state: EstimationState) -> dict:
        with logfire.span(f"node: {name}"):
            started = time.perf_counter()
            update = await fn(state)
            update.setdefault(
                "node_runs",
                [{"name": name, "duration_ms": (time.perf_counter() - started) * 1000}],
            )
            return update

    return wrapper
```

(Es el `with logfire.span(...)` del esqueleto del enunciado, subido a decorador para que ningún
nodo lo olvide y para que la duración llegue al acumulador `node_runs`. Si prefieres el `with`
explícito en cada nodo, vale; el criterio de "hecho" es el span por nodo.)

Los cinco nodos:

1. **`make_extract_requirements_node(llm_wrapper)`** — de la transcripción a una lista plana de
   requisitos atómicos. System prompt propio (analista que extrae requisitos concretos, uno por
   ítem, en inglés técnico, sin inventar lo que la transcripción no evidencia), llamada a
   `complete_structured` con `RequirementsExtraction` y
   `model_override=settings.graph_extraction_model`. El wrapper es síncrono: `asyncio.to_thread`.
   Devuelve `{"requirements": [...]}`.

2. **`make_classify_components_node(llm_wrapper)`** — agrupa los requisitos en componentes con
   categoría (`backend`, `integration`, `mobile`, `frontend`, ...). Misma mecánica con
   `ComponentClassification`. Devuelve `{"components": [...]}`.

3. **`make_search_budgets_node(backend)`** — por ahora **secuencial**: recorre `components`, y
   para cada uno consulta el backend inyectado con una query enfocada al componente
   (`"<name> (<category>)>"`). De cada ítem recuperado toma `estimated_hours`, `budget_id` y
   `distance`, y añade el `BudgetMatch`. Un fallo de recuperación de un componente es *soft*:
   `errors.append(...)` y se sigue con el siguiente (esto es lo único de manejo de errores que
   entra; lo serio va en el directo). Devuelve solo los matches **nuevos**: el reducer los
   concatena.

   El `backend` es un `BudgetSearchBackend` — el mismo protocolo de S12. No hay ni una línea de
   retrieval reimplementada aquí.

4. **`make_generate_estimate_node(...)`** — determinista: consolida `components` +
   `budget_matches` en un `EstimateReport` usando el modelo de costes de S12 (mediana +
   contingencia, `unbudgeted` a 0 h). Sin LLM. Atención al emparejado por nombre: los nombres de
   componentes vienen del LLM de clasificación; normaliza (minúsculas, strip de sufijos `[...]`)
   antes de buscar las referencias de cada componente.

5. **`make_validate_and_consolidate_node()`** — los guardrails de S12 portados a función pura
   sobre el estado: componente sin referencias → issue; horas fuera del rango plausible de sus
   referencias (×0.5 – ×2) → issue; total incoherente con la suma de componentes → issue. Fija
   `status = "validated"` si no hay issues, `"needs_review"` si hay, y los añade a `errors`.

Todos los logs con structlog y el `request_id` de `stage_logging` (`graph.node.search_budgets`,
...) en inglés, igual que el resto del servicio.

### Paso 4 — Checkpointer y observabilidad

`estimation_graph/checkpointer.py`:

1. **`saver_conninfo(settings) -> str`** — el `AsyncPostgresSaver` habla psycopg3 y quiere un DSN
   libpq plano: pelea `database_url` quitando el token de driver (`+psycopg`, `+asyncpg`). Es el
   espejo de `_async_database_url` en `app/foundation/persistence/database.py`, que hace el swap
   contrario.

2. **`open_checkpointer()`** — `@asynccontextmanager` que abre el
   `AsyncPostgresSaver.from_conn_string(conninfo)`, hace `await checkpointer.setup()` (idempotente:
   crea las tablas la primera vez, no-op después) y lo cede. El cuerpo de `setup()` y el cierre de
   conexión van una vez por arranque, nunca por petición.

Observabilidad (en `checkpointer.py`, en un `observability.py` del paquete, o en `main.py` — un
solo punto, documentado):

3. **`configure_logfire(app=None) -> bool`** — `logfire.configure(service_name=...,
   send_to_logfire="if-token-present", console=False)` + `logfire.instrument_fastapi(app)` si hay
   app + `logfire.instrument_httpx()` (captura las llamadas del SDK de OpenAI, que va sobre httpx).
   Guardado con flag de "configurado una vez" y try/except: la observabilidad nunca rompe el
   arranque. Sin token, no-op de exportación.

### Paso 5 — Cableado, compilación y lifespan

`estimation_graph/build.py`:

```python
def build_graph(
    *,
    llm_wrapper: LLMWrapper,
    backend: BudgetSearchBackend,
    checkpointer=None,
):
    builder = StateGraph(EstimationState)
    builder.add_node("extract_requirements", make_extract_requirements_node(llm_wrapper))
    builder.add_node("classify_components", make_classify_components_node(llm_wrapper))
    builder.add_node("search_budgets", make_search_budgets_node(backend))  # sequential for now
    builder.add_node("generate_estimate", make_generate_estimate_node(...))
    builder.add_node("validate_and_consolidate", make_validate_and_consolidate_node())
    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")
    builder.add_edge("validate_and_consolidate", END)
    return builder.compile(checkpointer=checkpointer)
```

**Nivel 3 (opcional):** sustituye la última arista por
`builder.add_conditional_edges("validate_and_consolidate", route_on_status,
{"validated": END, "needs_review": END})`, donde `route_on_status` lee el `status` y devuelve la
rama. Por ahora **ambas terminan en END**: la rama queda explícita, el reintento/fallback/HITL se
monta en el directo. Nada más.

En `app/main.py` (lifespan): `configure_logfire(app)` al arranque; dentro del lifespan,
`async with open_checkpointer() as cp:` construye el grafo con `get_llm_wrapper()` y
`make_budget_search_backend(top_k=settings.graph_search_top_k, distance_threshold=
settings.graph_search_distance_threshold)` y guárdalo en `app.state.graph`. Si `get_llm_wrapper()`
o el backend no están disponibles, `app.state.graph` queda a `None` y el endpoint responde 503:
mismo contrato de degradación del repo, sin críos el arranque.

### Paso 6 — Endpoint

`app/api/graph_estimation.py`, registrado en `app/main.py` con `prefix="/api/v1"`. Postura idéntica
a la del endpoint autónomo del agente S12:

- `POST /graph/estimate/from-transcript` con body `GraphEstimateRequest`, rate limit 10/min.
- `graph = getattr(request.app.state, "graph", None)`; `None` → 503 genérico.
- `estimation_id = payload.estimation_id or str(uuid4())`;
  `config = {"configurable": {"thread_id": estimation_id}}`.
- `result = await graph.ainvoke({"transcript": ..., "estimation_id": ...}, config)` dentro de un
  `log_stage("graph_estimate", ...)` para correlacionar con el `request_id`. Cualquier excepción →
  502 con mensaje genérico.
- Devuelve `GraphEstimateResponse` con los campos del estado (`status`, `estimate`,
  `requirements`, `components`, `budget_matches`, `node_runs`, `errors`) **más** `run_text`: el
  resumen renderizado en servidor (pipeline de nodos con duraciones → componentes → matches →
  estimación → status), mismo patrón que `trace_text` en S12. La UI lo pinta y lo descarga.

El resto del sistema no se entera: `boss`, `critic`, el agente S12 y el pipeline RAG siguen igual.

### Paso 7 — Script CLI

`scripts/run_graph_s13.py`, siguiendo la convención de `scripts/run_agent_s12.py`:

- `--transcript PATH` (default `examples/session-12/sample_transcript_complex.txt`).
- `--estimation-id ID` (default derivado del nombre del fichero; es el `thread_id`).
- `--memory`: `MemorySaver` en proceso en lugar del checkpointer de Postgres (sin DB para el
  estado; los nodos LLM siguen necesitando clave).
- `--stub`: reutiliza `examples/session-12/offline_budget_stub.py` como backend de recuperación
  (sin Docker, sin Postgres para la búsqueda).
- `--out PATH`: escribe la ejecución renderizada (default
  `examples/session-13/trace_complex.txt`).

**Disciplina de coste** (documentada en el docstring del script): primero la mecánica con
`--memory --stub` sobre la transcripción **simple** (gasta dos llamadas a `gpt-5-mini`); la
ejecución de entrega es el camino real (checkpointer de Postgres + retrieval real) sobre la
**compleja**, y su traza se guarda con `--out examples/session-13/trace_complex.txt` y se copia al
documento de resultados. Si tienes `LOGFIRE_TOKEN`, la misma ejecución deja la traza exportada con
un span por nodo: enlázala en el doc.

### Paso 8 — UI

En `cliente-web/`, replicando el patrón de `agent-estimations` de S12:

- `src/graphEstimationClient.js` — `HttpGraphEstimationClient` con el mismo estilo que
  `agentEstimationClient.js`: `fetch` nativo con timeout (`GRAPH_TIMEOUT_MS`, default 300000: el
  grafo son dos llamadas LLM + retrieval, más corto que el bucle del agente), cabecera `X-API-Key`
  si aplica, clase de error propia, `#post` que parsea el `detail` de FastAPI. Método
  `estimateFromTranscript(transcript, estimationId?)` → `POST /api/v1/graph/estimate/from-transcript`.
- `src/routes/graphEstimations.js` — `graphEstimationsRouter({ graphClient })`:
  `GET /graph-estimations/new` (formulario con la transcripción, validación de longitud mínima
  como en S12) y `POST /graph-estimations` (ejecuta y renderiza; errores del cliente re-renderizan
  el formulario con el mensaje).
- `views/graph-estimations/` — `new.ejs` (textarea de transcripción, mismo patrón `.field` /
  `.btn-primary`), `show.ejs` (cabecera con `estimation_id` y badge de `status`, includes de los
  parciales), `_pipeline.ejs` (timeline: un bloque por `NodeRun` con nombre del nodo, duración en
  ms y los matches/errores que aportó) y `_estimate.ejs` (componentes con horas, referencias y
  rationale; total; issues de `errors`).
- Botón **Descargar ejecución (.txt)** que baja el `run_text` tal cual (blob client-side, sin
  endpoint extra).
- Enlace en `views/partials/header.ejs` y montaje condicional en `src/app.js`
  (`if (graphClient) ...`), con el cliente construido en `src/server.js` desde
  `config.estimatorApiBaseUrl`. Si tocas clases nuevas, relanza `npm run build:css` (Tailwind
  escanea los `.ejs`).

### Paso 9 — Tests

Planos, con dobles, sin red ni Postgres:

- `tests/test_graph_state.py` — los reducers: dos actualizaciones parciales sobre
  `budget_matches` se concatenan; `errors` igual; `node_runs` igual. Y `total=False`: un dict
  parcial no exige todas las claves.
- `tests/test_graph_nodes.py` — con un `LLMWrapper` falso (respuesta estructurada guionizada) y un
  `BudgetSearchBackend` falso:
  - `extract_requirements` / `classify_components` devuelven exactamente lo que el modelo
    estructurado dice, sin claves extra.
  - `search_budgets` devuelve solo los matches nuevos, filtra ítems sin `estimated_hours`, y un
    componente cuyo backend lanza excepción acaba en `errors` sin romper el nodo.
  - `generate_estimate` es determinista: misma entrada, misma salida; mediana + contingencia;
    componente sin referencias → `unbudgeted`, 0 h y fuera del total; el emparejado por nombre
    sobrevive a drift de mayúsculas/sufijos.
  - `validate_and_consolidate` fija `status="validated"` limpio y `"needs_review"` con issues.
- `tests/test_graph_build.py` — el grafo entero con `MemorySaver` y dobles: corre de START a END,
  el estado final lleva los cinco `node_runs` en orden, y (si hiciste el Nivel 3)
  `route_on_status` devuelve la rama correcta según `status`. Dos `ainvoke` con `thread_id`
  distinto no comparten estado.
- `tests/test_graph_endpoint.py` — con dependency overrides / app de prueba: `app.state.graph` a
  `None` → 503; grafo falso → 200 con cuerpo coherente (`status`, `estimate`, `node_runs`,
  `run_text`).
- `cliente-web/test/graphEstimations.test.js` — node:test + supertest contra `createApp` con un
  `fakeClient()`: GET del formulario, transcripción corta → re-render con error, ejecución
  correcta → 200 con la timeline y el botón de descarga.

Y `tests/test_layering.py` debe seguir en verde **sin tocarlo**: esa es la prueba de que la
inyección del grafo está bien hecha (los nodos no importan `app.dependencies`; el paquete no
importa `app.generation.rag`).

### Paso 10 — Documento de resultados

`docs/sesion-13-grafo-langgraph.md`, con:

1. Qué se construyó (paquete, grafo, checkpointer, observabilidad, endpoint, UI) y qué se
   reutilizó de S9–S12.
2. La **traza real** de la ejecución de entrega sobre `sample_transcript_complex.txt` (la misma
   que `examples/session-13/trace_complex.txt`): pipeline de nodos con duraciones, requisitos →
   componentes, matches por componente, estimación y `status` final.
3. Evidencia de persistencia: el `thread_id` usado y una consulta a las tablas del checkpointer
   (`checkpoints` / `checkpoint_writes`) mostrando que el estado de esa ejecución vive en el
   Postgres del proyecto.
4. La traza de Logfire (captura o enlace) con un span por nodo — o, si no configuraste token, una
   nota explícita de que la timeline vive en `node_runs` y cómo activarla.
5. Una nota de 2–3 frases sobre el coste/tiempos por nodo: qué nodo domina el tiempo total (será
   el punto de partida del directo sobre paralelismo).

## Entregables

- `app/generation/agentic/estimation_graph/` (`state.py`, `schemas.py`, `nodes.py`, `build.py`,
  `checkpointer.py` [+ `observability.py` opcional]).
- Settings del grafo y de Logfire en `app/config.py`; wiring y compilación en el lifespan de
  `app/main.py`.
- `app/api/graph_estimation.py` con `POST /api/v1/graph/estimate/from-transcript` y su registro.
- `scripts/run_graph_s13.py`.
- `examples/session-13/trace_complex.txt` con la traza de la ejecución de entrega.
- UI en `cliente-web/` (cliente, rutas, vistas, header, descarga) y su test.
- `tests/test_graph_state.py`, `tests/test_graph_nodes.py`, `tests/test_graph_build.py`,
  `tests/test_graph_endpoint.py`.
- `docs/sesion-13-grafo-langgraph.md` con la evidencia real.

## Criterios de aceptación

Con `sample_transcript_complex.txt`, camino real (Postgres + retrieval), en orden:

- [ ] El grafo corre de principio a fin: los cinco nodos se ejecutan en orden y el endpoint
      devuelve la estimación con su `status`; el contrato hacia el backend de negocio es el de
      siempre (transcripción → estimación estructurada).
- [ ] El estado es tipado (`TypedDict`) y tiene **al menos un** reducer acumulador funcionando
      (`budget_matches` crece a lo largo del flujo; los tests lo demuestran).
- [ ] Los cinco nodos son funciones que reciben el estado y devuelven actualizaciones parciales;
      ninguna devuelve el estado entero reconstruido.
- [ ] El checkpointer persiste en el Postgres del proyecto y cada ejecución lleva su `thread_id`
      (el `estimation_id`); las tablas del checkpointer coexisten con las de embeddings.
- [ ] Existe una **traza completa** de una ejecución sobre `sample_transcript_complex.txt`, con un
      span por nodo (Logfire exportado o timeline de `node_runs` + spans locales), guardada en
      `examples/session-13/trace_complex.txt` y reproducida en el doc.
- [ ] La transcripción compleja produce **más de un componente** y `budget_matches` acumula
      matches de varios componentes (el caso del enunciado).
- [ ] `uv run pytest`, `ruff` y `mypy` en verde, incluido `tests/test_layering.py` sin cambios; y
      `npm test` en `cliente-web` en verde.

## Checklist de verificación

- [ ] `estimation_graph/` no importa `app.dependencies` ni `app.generation.rag` (lo comprueba
      `test_layering.py`); el wiring vive en el composition root y el lifespan.
- [ ] Ningún retrieval reimplementado dentro del grafo: `search_budgets` delega en el
      `BudgetSearchBackend` inyectado (construido con `make_budget_search_backend`).
- [ ] El grafo trabaja en engineer-hours de punta a punta; no hay ninguna conversión ÷8.
- [ ] El DSN del checkpointer es un DSN libpq plano pelado desde `database_url`; no un URL de
      SQLAlchemy.
- [ ] `await checkpointer.setup()` se llama una vez por arranque (idempotente), nunca por petición.
- [ ] El grafo se compila una vez por arranque y vive en `app.state.graph`; el endpoint responde
      503 si no está.
- [ ] Logfire configurado con `if-token-present`: sin token todo funciona; con token, un span por
      nodo + FastAPI + httpx instrumentados.
- [ ] Un `thread_id` repetido **reanuda** el hilo del checkpointer: cada ejecución de entrega usa
      un `estimation_id` fresco (uuid4) salvo que quieras reanudar a propósito.
- [ ] Todo el código, docstrings, prompts, nombres de nodos y logs están en inglés; los spans
      siguen el patrón `node: <name>`.
- [ ] La UI replica el patrón S12 (cliente, rutas, vistas con parciales, descarga) y `npm run
      build:css` está relanzado si se añadieron clases.
- [ ] `docs/sesion-13-grafo-langgraph.md` contiene la traza real y la evidencia de persistencia,
      no ejemplos inventados.
- [ ] La rama es `feature/pre-session-13` y el PR apunta a `main`.

## Documentación de referencia

- Spec de la Sesión 12 (capa agéntica con bucle manual):
  [`sesion-12-capa-agentica-bucle-manual.md`](./sesion-12-capa-agentica-bucle-manual.md)
- Documento de resultados de la Sesión 12:
  [`../docs/sesion-12-capa-agentica.md`](../docs/sesion-12-capa-agentica.md)
- LangGraph — StateGraph, nodos, aristas y estado:
  https://docs.langchain.com/oss/python/langgraph
- LangGraph — persistencia y checkpointers (Postgres):
  https://docs.langchain.com/oss/python/langgraph/persistence
- Logfire — primeros pasos e instrumentación:
  https://pydantic.dev/docs/logfire/get-started/
- LangSmith (alternativa de trazas, si optaras por ella):
  https://docs.smith.langchain.com/

## Nota

**Qué es nuestro (y por qué).** El repo de referencia resuelve la sesión con un paquete
`app/domain/graph/` cuyos nodos se auto-cablean importando `app.dependencies`, una tercera llamada
al LLM para consolidar la estimación, unidades en engineer-days (dividiendo las horas del corpus
entre 8) y sin UI. Aquí nada de eso encaja: `test_layering.py` prohíbe esos imports fuera del
composition root (los nodos son clausuras con `LLMWrapper` y `BudgetSearchBackend` inyectados, el
grafo se compila en el lifespan), S12 fijó engineer-hours y el modelo de costes determinista ya
existe en `estimation_agent/tools.py` (usarlo de nuevo es gratis y ancla las cifras en las
referencias históricas en vez de en un tercer arbitrio del LLM), y la UI de trazas ya tiene patrón
propio desde S12. De la referencia se toma la forma del estado tipado del enunciado y la pauta
"un span por nodo"; el resto se escribe sobre los contratos de este repo.

**Lo que NO hay que hacer.** No paralelices `search_budgets`: la Send API, medir el coste de lo
secuencial y paralelizar es el ejercicio del directo; si lo traes ahora, el directo no tiene
carne. No montes reintentos, fallbacks, timeouts por nodo ni `interrupt()`: un fallo soft de un
componente apunta a `errors` y sigue. No reimplementes el retrieval ni el modelo de costes dentro
de los nodos. No conviertas horas a días: el grafo hereda las unidades del corpus y de S12. Y no
compiles el grafo ni hagas `setup()` por petición: una vez por arranque.

**La trampa de la sesión.** Hay tres, y las tres son silenciosas. La primera: `AsyncPostgresSaver`
quiere un DSN libpq plano; si le pasas el `postgresql+psycopg://` de `database_url` tal cual, la
conexión no es lo que crees — pelar el token de driver es un paso que el test de `saver_conninfo`
existe para blindar. La segunda: reutilizar un `thread_id` terminado. El checkpointer **reanuda**
hilos: invocar dos veces con el mismo `estimation_id` puede concatenar ejecuciones en los
acumuladores (`budget_matches` duplicados, `status` heredado) sin ningún error visible. Cada
ejecución lleva un `estimation_id` fresco, salvo reanudación deliberada — el test de aislamiento
por `thread_id` existe para eso. Y la tercera: olvidar el `await checkpointer.setup()` antes del
primer `ainvoke` — el crash llega tarde (en la primera persistencia), no al arrancar, así que el
smoke test del endpoint no lo caza si no hay una ejecución real detrás.
