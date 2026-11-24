# Spec — Proyecto 6: Capa agéntica de estimación: bucle manual con tools (Sesión 12)

## Objetivo

El estimador funciona hoy con un pipeline **fijo**: reformular la transcripción, recuperar
presupuestos, generar la estimación. Ese orden es siempre el mismo porque alguien lo decidió por
adelantado. Pero una transcripción real no siempre encaja en un único camino: un proyecto que mezcla
un backend de negocio, una integración con un ERP y una app móvil obliga a buscar presupuestos
históricos **por separado** para cada componente, calcular estimaciones parciales y consolidar. Ni
el número de búsquedas ni su orden se conocen de antemano: los decide lo que diga la transcripción.

Esta sesión añade la pieza que falta: una **capa agéntica** que decide qué hacer en cada paso. Y la
construye a mano, sin framework, para que quede a la vista de qué está hecho un agente: **un bucle
que llama a un LLM que decide, ejecuta tools y para cuando ha terminado**.

Al terminar tendrás:

- Un paquete `app/generation/agentic/estimation_agent/` con tres piezas: schemas (argumentos
  validados, traza, resultado), tools (schemas planos para la Responses API + funciones Python) y el
  runner (el bucle razona → actúa → observa → repite).
- Dos tools que envuelven infraestructura que ya existe: `search_budgets` reutiliza el pipeline de
  recuperación híbrida + reranking de las Sesiones 9–10; `calculate_estimate` es aritmética
  determinista. Ni una ni otra se reimplementan.
- Una tercera tool, `validate_estimate`, que aplica los guardrails de la Sesión 4 sobre la
  estimación final antes de devolverla (el opcional del enunciado entra como pieza obligatoria).
- Un bucle **conducido a mano** sobre la Responses API (`client.responses.create`, `gpt-5`,
  `reasoning={"effort": "medium"}`): recorrido de `response.output`, ejecución de cada
  `function_call`, devolución de `function_call_output` con su `call_id`, encadenado con
  `previous_response_id` y condición de parada natural más un máximo de iteraciones.
- Una traza paso a paso (razonamiento + acción + observación por iteración) visible en la UI y
  descargable como `.txt`.
- Un endpoint nuevo, `POST /api/v1/agent/estimate/from-transcript`, y un script CLI
  (`scripts/run_agent_s12.py`) para depurar el bucle barato antes de la ejecución de entrega.

El backend de negocio no ve el bucle: sigue enviando una transcripción y recibiendo una estimación
estructurada, igual que en la Sesión 9. La diferencia es que ahora, junto a la estimación, llega la
traza de cómo se decidió.

> **Alcance estricto.** Esta sesión **no** toca el pipeline RAG de S9–S11, ni el retrieval, ni el
> wizard, ni la base de datos (no hay migraciones ni persistencia de ejecuciones: cada llamada al
> agente es un bucle fresco y sin efectos colaterales). Tampoco introduce un framework de agentes:
> el bucle se escribe a mano y esa es justo la mitad del ejercicio.

## Reconciliación con el repo

El punto de partida es `main` con la Sesión 11 cerrada: citación verificable por línea, gate de
calidad, augmentación, síntesis y baseline RAGAS. El repo de referencia
(`LIDR-academy-temp/ai-engineering`, rama `session_12`) sirve de **guía de alcance**, no de
plantilla: sus transcripciones de prueba y su esqueleto de cálculo se reutilizan; su código de
agente, sus nombres y su estructura de paquete no. Todo lo que sigue se diseña sobre los contratos
y convenciones de este repo.

**Decisiones estructurales propias:**

- El paquete vive en `app/generation/agentic/estimation_agent/`, junto al `boss.py` y `critic.py`
  que ya existen, y se reparte en `schemas.py`, `tools.py` y `runner.py`. No se replica la
  estructura de la referencia (`agent_loop.py` / `agent_schemas.py` / `agent_tools.py`): son tres
  nombres para un solo agente y aquí el paquete ya tenía dueño.
- **El layering manda y la referencia lo incumple.** En el repo de referencia, `agent_tools.py`
  importa `retrieve()` y `get_embedder` directamente. Aquí `tests/test_layering.py` lo prohíbe:
  `agentic` no puede importar a su hermano `generation.rag`, y solo el composition root puede
  importar `app.dependencies`. La solución es la que ya usa el repo para el mismo problema:
  `tools.py` declara un **protocolo** `BudgetSearchBackend` y las funciones puras; el backend
  concreto que envuelve `retrieve()` se construye en `app/dependencies.py`
  (patrón `get_semantic_retriever`). El runner recibe cliente y backend **inyectados**.
- **Excepción deliberada y documentada a `LLMWrapper`.** Todo el resto del repo llama al LLM a
  través de `LLMWrapper` (LiteLLM + Instructor). El bucle del agente habla directamente con la
  Responses API de OpenAI (`client.responses.create` / `.parse`) porque manejar a mano los items
  `function_call` / `function_call_output` y el encadenado por `previous_response_id` es el
  ejercicio. La excepción se documenta en el docstring del runner, como aviso para quien venga
  detrás: no lo "arregles".
- La configuración va en `app/config.py` con nuestro snake_case (no las mayúsculas de la
  referencia): `agent_model`, `agent_reasoning_effort`, `agent_max_iterations`,
  `agent_search_top_k`, `agent_search_distance_threshold`.
- **Unidades: engineer-hours.** El corpus registra horas (`estimated_hours` en los chunks y en
  `data/task_corpus.json`); las tools trabajan en horas de punta a punta, sin conversiones. El
  pipeline RAG de S9–S11 sigue en engineer-days: son caminos distintos y no se mezclan.
- `search_budgets` busca sobre `chunk_types=["budget_component", "historical_task"]` de la
  colección `budget`. Ambos tipos llevan `estimated_hours` en metadata; restringir solo a
  `historical_task` (como hace la referencia) dejaría fuera los componentes de presupuesto, que son
  justo las referencias análogas que un componente de transcripción necesita.
- El cliente async (`get_async_openai_client`) devuelve `None` sin clave configurada y el router lo
  traduce a un 500 con mensaje genérico, igual que `get_embedder` y `get_semantic_retriever`.
- El material de la sesión vive en `examples/session-12/` (no en `exercises/`), los tests son planos
  en `tests/test_agent_*.py` (convención del repo, sin subdirectorios) y el servicio de Compose se
  llama `api`.

**Material tomado tal cual del repo de referencia** (se copia a `examples/session-12/`):

| Archivo de la referencia | Destino | Papel |
|---|---|---|
| `exercises/session-12/sample_transcript_simple.txt` | `examples/session-12/sample_transcript_simple.txt` | Transcripción de un componente; depuración barata del bucle. |
| `exercises/session-12/sample_transcript_complex.txt` | `examples/session-12/sample_transcript_complex.txt` | Transcripción multicomponente; input de los criterios de aceptación. |
| `exercises/session-12/calculate_estimate_skeleton.py` | `examples/session-12/calculate_estimate_skeleton.py` | Esqueleto del modelo de costes determinista: se completa y se integra, no se rediseña. |

**Lo que se escribe propio** (y por qué no vale copiarlo): todo el código del agente, el system
prompt, las descripciones de las tools, el stub offline de recuperación, el endpoint, la UI y el
script CLI. La traza de ejemplo de la referencia no se copia: el entregable es la traza de
**nuestra** ejecución.

### Piezas que se reutilizan

- `app/generation/rag/retrieval/pipeline.py` — `retrieve()`: híbrido + reranking de S10.
  `search_budgets` lo invoca; no lo reconstruye.
- `app/generation/rag/schemas.py` — `RetrievedChunk` (con `sector`, `budget_id`,
  `estimated_hours`, `distance`) y `Collection`.
- `app/generation/rag/embedding/embedder.py` — `OpenAIEmbedder.embed_one` (mismo modelo que la
  ingesta: `text-embedding-3-small`).
- `app/foundation/llm/runtime_config.py` — `RuntimeRetrievalConfig`: el backend hereda el
  `search_mode`/`rerank` que la UI de Ajustes tiene configurado en ese momento.
- `app/generation/rag/stage_logging.py` — `current_request_id`: los logs del bucle quedan
  correlacionados con la petición HTTP igual que los del pipeline RAG.
- `app/dependencies.py` — patrón de constructores `... | None` con degradación a 500 en el router.
- `app/foundation/guardrails/` — la *idea* de los guardrails de S4; `validate_estimate` es una
  función pura nueva, no un import de los existentes (estos operan sobre `GroundedEstimate`, un
  contrato distinto).
- `cliente-web/src/routes/ragEstimations.js` + `views/rag-estimations/` — patrón de rutas EJS y
  vistas por pasos que replica la vista de trazas del agente.
- `tests/test_hybrid_retrieval.py`, `tests/test_rag_stages.py` — estilo de tests con dobles, sin
  Postgres ni red.

## Requisitos para el ejercicio

- Proyecto de la Sesión 11 en verde:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy app frontend
  ```

- Stack levantado desde `estimador-cag/`:

  ```bash
  docker compose up -d --build
  curl -s http://localhost:8000/health | jq
  ```

- Corpus ingerido de forma idempotente (los dos corpus: presupuestos y tareas):

  ```bash
  docker compose run --rm api python scripts/query_examples.py
  docker compose run --rm api python scripts/build_task_corpus.py --ingest
  ```

  El agente busca sobre `budget_component` **y** `historical_task`, así que conviene tener ambos.

- Clave de OpenAI en `.env` con acceso a la Responses API y a `gpt-5` (`OPEN_AI_KEY` o la variable
  que use tu `.env` para el proveedor OpenAI). Sin clave no hay bucle: el router devolverá 500.

- El paquete `openai` ya está en `pyproject.toml` (`>=3.13.0`) y soporta `client.responses.create`.
  No se añade ninguna dependencia nueva.

## Lo que entra y lo que no entra

**Entra:**

- `app/generation/agentic/estimation_agent/` con `schemas.py`, `tools.py` y `runner.py`.
- Las tres tools (`search_budgets`, `calculate_estimate`, `validate_estimate`) con schema plano
  Responses y `strict: true`, nombres/descripciones/parámetros en inglés.
- El bucle manual con `previous_response_id`, captura de reasoning summaries, salvaguarda de
  `max_iterations` y parseo final estructurado (`client.responses.parse`).
- `get_async_openai_client()` y `get_budget_search_backend()` en `app/dependencies.py`.
- Settings del agente en `app/config.py`.
- `POST /api/v1/agent/estimate/from-transcript` en `app/api/agent_estimation.py`.
- Script CLI `scripts/run_agent_s12.py` con modo `--stub` offline.
- Stub propio de depuración `examples/session-12/offline_budget_stub.py`.
- UI en `cliente-web/`: formulario de transcripción, ejecución, timeline de trazas paso a paso y
  descarga de la traza en `.txt`.
- Tests con dobles del cliente y del backend, sin red ni Postgres.
- Documento de resultados `docs/sesion-12-capa-agentica.md` con la traza real de la ejecución sobre
  `sample_transcript_complex.txt`.

**No entra:**

- Ningún framework de agentes (LangGraph, autogen, el modo agéntico interno de la Responses API).
  El encadenado lo conduce nuestro código.
- Cambios en el pipeline RAG, el retrieval, el wizard, el corpus o la base de datos.
- Persistencia de las ejecuciones del agente (ni tabla ni migración): cada ejecución es stateless.
- Reimplementar la recuperación dentro de `search_budgets`.
- Cambiar el contrato HTTP existente (`/estimate/from-transcript` y `/estimate/stages/*` siguen
  igual; el agente es un endpoint nuevo, no un reemplazo).
- Copiar el código del agente del repo de referencia: solo se toman las dos transcripciones y el
  esqueleto de cálculo listados arriba.

## Datos para el ejercicio

### Transcripciones

`examples/session-12/sample_transcript_simple.txt` y `sample_transcript_complex.txt`, tomadas del
repo de referencia. La simple describe un proyecto de un solo componente (para depurar la mecánica
del bucle gastando lo mínimo); la compleja mezcla varios componentes de naturaleza distinta —el
caso del enunciado: backend de negocio + integración + app— y es el input de los criterios de
aceptación.

### Modelo de costes

`examples/session-12/calculate_estimate_skeleton.py`, también de la referencia. Define el
comportamiento que `calculate_estimate` debe cumplir (los detalles exactos están en el esqueleto;
el resumen operativo):

- Por componente: **mediana** de los `reference_amounts` como estimación central, multiplicada por
  un **buffer de contingencia** fijo y transparente.
- Componente sin referencias: se marca como `unbudgeted` con 0 horas — **no se inventa un número**;
  el agente debe notarlo (vía `validate_estimate`) y volver a buscar.
- Total: suma de los componentes, redondeada a una decimal.

El Paso 2 consiste en completar el esqueleto y adaptarlo a nuestros modelos de argumento, no en
rediseñar el modelo de costes. El centro del ejercicio es el bucle.

### Stub offline de depuración

Nuestro pipeline de S9–S10 **está listo y funcionando**: ese es el camino principal y el único que
cuenta para la entrega. Pero depurar la mecánica del bucle (parseo de argumentos, `call_id`,
condición de parada) con `gpt-5-mini` y sin levantar Docker/Postgres es más rápido y barato. Para
eso se escribe `examples/session-12/offline_budget_stub.py`: un `BudgetSearchBackend` alternativo
que devuelve ítems históricos enlatados por similitud léxica trivial con la query. Es una
herramienta de depuración, no una red de seguridad del pipeline.

## Ejercicio paso a paso

### Paso 0 — Settings, cliente async y gate

1. Añade a `app/config.py`:

   ```python
   agent_model: str = "gpt-5"
   agent_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"
   agent_max_iterations: int = 10
   agent_search_top_k: int = 8
   agent_search_distance_threshold: float = 0.6
   ```

2. Añade `get_async_openai_client()` a `app/dependencies.py`. Mismo contrato de degradación que
   `get_embedder`: `AsyncOpenAI | None`, `None` sin clave. El cliente no reutiliza el de embeddings
   (base URL y timeouts distintos) ni el de `LLMWrapper` (síncrono, LiteLLM).

3. Añade `get_budget_search_backend()` a `app/dependencies.py`. Es el **único** sitio donde el
   agente toca el pipeline RAG: embebe la query con el embedder de la ingesta, resuelve el
   `RuntimeRetrievalConfig` vigente y llama a `retrieve()` sobre la colección `budget` con
   `chunk_types=["budget_component", "historical_task"]`, `top_k=agent_search_top_k` y
   `distance_threshold=agent_search_distance_threshold`. Devuelve la lista de dicts que
   `search_budgets` espera (ver Paso 2). Si no hay embedder, `None` y el router responde 500.

### Paso 1 — Schemas del agente

`app/generation/agentic/estimation_agent/schemas.py`. Tres familias de modelos Pydantic, todos con
descripciones en inglés:

1. **Argumentos de tools.** `SearchFilters` (`sectors`, `component_type`, ambos opcionales),
   `SearchBudgetsArgs`, `ComponentCostInput` (`name`, `reference_amounts`),
   `CalculateEstimateArgs`, `ValidationLine` (`name`, `estimated_hours`, `reference_amounts`),
   `ValidateEstimateArgs`. El runner valida cada `json.loads(call.arguments)` contra el modelo
   **antes** de ejecutar la tool: un argumento alucinado se convierte en un error que el modelo
   puede corregir en la siguiente iteración, nunca en una excepción que reviente el bucle.

2. **Traza.** `AgentStepRecord` (`step`, `reasoning_summary`, `tool`, `tool_args`, `observation`) y
   `AgentTrace` con `render()`, que produce el formato de consola mínimo que exige el enunciado:

   ```text
   STEP 1
     reasoning:   <what the agent decided and why>
     action:      search_budgets({"query": "...", "filters": {...}})
     observation: <summary of returned items>
   ```

   El render es nuestro y algo más rico que el mínimo (cabecera con modelo/iteraciones y cola con
   el motivo de parada), pero cada paso mantiene exactamente el triplete
   razonamiento + acción + observación.

3. **Resultado.** `AgentComponent` (`name`, `estimated_hours`, `source_chunk_ids`, `rationale`),
   `AgentEstimateReport` (`components`, `total_hours`, `assumptions`, `confidence`),
   `AgentRunResult` (`report`, `trace`, `iterations`, `stopped_reason`) con
   `stopped_reason: Literal["completed", "max_iterations", "no_final_report"]`.

   Es deliberadamente **ligero**: no es `GroundedEstimate` (que exige citas verificadas por línea y
   chequeos de coherencia). El agente produce su propio informe con los chunk ids que respaldan
   cada componente; si mañana hay que fusionarlo con el contrato pesado, será un ejercicio de
   adaptación, no un cambio en esta capa.

### Paso 2 — Tools

`app/generation/agentic/estimation_agent/tools.py`. Cuatro bloques:

1. **Schemas planos para la Responses API.** Lista `TOOL_SCHEMAS` con la forma
   `{"type": "function", "name": ..., "description": ..., "parameters": {...}, "strict": true}`.
   Recuerda la trampa: en Responses, `type` y `name` van **al mismo nivel**; no anides bajo una
   clave `function` como en Chat Completions. Con `strict: true`, todas las propiedades del schema
   deben aparecer en `required` (la opcionalidad se modela con unions `["object", "null"]`) y
   `additionalProperties: false` en todos los niveles.

   Las **descripciones son lo único que el modelo lee** para decidir cuándo usar cada tool:
   escríbelas para un modelo que no ve tu código. La de `search_budgets` debe dejar claro que se
   llama **una vez por componente** y con una query enfocada a ese componente (no al proyecto
   entero); la de `calculate_estimate`, que es aritmética pura y reproducible y que espera las
   horas que devolvió `search_budgets`; la de `validate_estimate`, que es el **último** paso antes
   de responder y que hay que subsanar lo que flaggee.

2. **Protocolo del backend.** `BudgetSearchBackend`: un callable async que recibe los
   `SearchBudgetsArgs` validados y devuelve `list[dict]`. Nada más: la implementación real vive en
   `app/dependencies.py` (Paso 0.3) y el stub de depuración en `examples/session-12/`.

3. **Implementaciones.** `search_budgets(raw_args, backend)` (valida, delega en el backend,
   devuelve ítems + resumen legible), `calculate_estimate(raw_args)` (completa el esqueleto:
   mediana + contingencia, `unbudgeted` a 0 h) y `validate_estimate(raw_args)` (guardrails S4:
   componente sin referencias, estimación fuera del rango plausible de sus referencias, total
   incoherente con la suma de componentes, total no positivo o desproporcionado). Las dos últimas
   no tocan el LLM: son deterministas y se pueden probar con pytest puro.

4. **Dispatch.** `dispatch_tool(name, raw_args, backend)` enruta a la implementación. Tool
   desconocida → `ValueError`; cualquier excepción (incluida una `ValidationError` de argumentos)
   la captura el runner y la devuelve al modelo como observación de error.

### Paso 3 — El bucle

`app/generation/agentic/estimation_agent/runner.py`. La función principal:

```python
async def run_estimation_agent(
    transcript: str,
    *,
    client: Any,  # AsyncOpenAI, inyectado
    model: str,
    reasoning_effort: str = "medium",
    max_iterations: int = 10,
    backend: BudgetSearchBackend,
) -> AgentRunResult: ...
```

Y el system prompt, escrito por nosotros, con rol y método explícitos: descomponer la transcripción
en componentes, buscar presupuestos **por cada uno**, calcular la estimación parcial y consolidar,
validar con `validate_estimate` como último paso y parar cuando esté satisfecho. Que sepa qué tools
tiene y que fundamente las cifras en lo que devuelva `search_budgets`, registrando como suposición
todo lo que la transcripción no especifique.

Mecánica vuelta a vuelta:

1. **Arranque.** `client.responses.create(model=..., instructions=SYSTEM_PROMPT,
   input=[{"role": "user", "content": transcript}], tools=TOOL_SCHEMAS,
   reasoning={"effort": ..., "summary": "auto"}, store=True)`.
2. **Extracción.** Recorre `response.output` buscando items `function_call` (con `call_id`, `name`
   y `arguments`). Si no hay ninguno, el modelo ha terminado: salida natural del bucle.
3. **Ejecución.** Por cada `function_call`: parsea `arguments` desde JSON, valida contra el modelo
   del Paso 1, ejecuta con `dispatch_tool` y registra en la traza el razonamiento del turno (los
   reasoning summaries de la API; puede que vengan vacíos con efforts baratos — el paso queda con
   `(no reasoning summary emitted)`), la acción y la observación. Si el turno trae **varias** calls
   en paralelo, el summary pertenece al turno: se cuelga del primer paso y los hermanos se marcan
   como calls paralelas, sin repetir el bloque.
4. **Respuesta.** Un `function_call_output` por call, cada uno con **su** `call_id` y el resultado
   serializado. El `call_id` es la parte que más se olvida: sin él la API no puede emparejar
   resultado con llamada.
5. **Encadenado.** Re-llamada con `previous_response_id=response.id` e `input=tool_outputs` (el
   servidor conserva el razonamiento y las calls previas con su orden). **No** delegues el bucle al
   comportamiento agéntico interno de la Responses API: condúcelo a mano, que capturar y mostrar
   cada paso es medio ejercicio.
6. **Parada.** Repite mientras haya `function_call`. Salvaguarda: `max_iterations` (por defecto 10,
   configurable); si se alcanza, `stopped_reason="max_iterations"` y se devuelve lo que haya con
   `report=None`.
7. **Cierre.** Una última llamada `client.responses.parse(..., previous_response_id=response.id,
   text_format=AgentEstimateReport)` con una instrucción final de consolidación convierte el
   contexto acumulado en el informe estructurado. Si el parseo falla, `stopped_reason=
   "no_final_report"`: es un motivo de parada, no un crash.

Todos los logs con structlog y `request_id` del `stage_logging` (`agent.run.start`,
`agent.tool.search_budgets`, `agent.run.done`, ...), en inglés.

### Paso 4 — Script CLI

`scripts/run_agent_s12.py`, siguiendo la convención de `scripts/demo_verify_citations_s11.py`:
parsea argumentos, carga el transcript, ejecuta el runner e imprime traza + informe. Banderas:

- `--model` (default `settings.agent_model`) y `--effort`.
- `--max-iterations`.
- `--stub`: usa el backend offline de `examples/session-12/offline_budget_stub.py` en lugar del
  pipeline real (sin Docker, sin base de datos).
- `--out PATH`: escribe la traza renderizada + el informe en un `.txt`.

**Disciplina de coste** (documentada en el docstring del script): primero depura la mecánica del
bucle con `--model gpt-5-mini --effort minimal` sobre la transcripción **simple** (y `--stub` si no
quieres levantar la pila); la ejecución de entrega es `gpt-5` / `medium` sobre la **compleja**, con
el pipeline real, y su traza se guarda con `--out examples/session-12/trace_complex.txt`.

### Paso 5 — Endpoint

`app/api/agent_estimation.py`, registrado en `app/main.py` con `prefix="/api/v1"`:

- `POST /agent/estimate/from-transcript` con body `{"transcript": str}`.
- Resuelve `get_async_openai_client()` y `get_budget_search_backend()`; cualquiera de ellos a
  `None` → 500 con mensaje genérico (patrón del repo).
- Ejecuta `run_estimation_agent` y devuelve el `AgentRunResult` serializado, **más** un campo
  `trace_text` con el `AgentTrace.render()` ya renderizado en servidor (la UI lo pinta y lo
  descarga sin duplicar el formato en JavaScript).
- Sin persistencia ni idempotencia: cada POST es una ejecución fresca. El bucle puede tardar varias
  llamadas al LLM; el endpoint es síncrono como `/estimate/from-transcript` y eso se documenta en
  el docstring.

El resto del sistema no se entera: `boss`, `critic` y el pipeline RAG siguen igual.

### Paso 6 — UI

En `cliente-web/`, replicando el patrón de `rag-estimations`:

- `src/agentEstimationClient.js` — cliente del endpoint (mismo estilo que
  `ragEstimationClient.js`).
- `src/routes/agentEstimations.js` — `GET /agent-estimations/new` (formulario con la transcripción,
  precargada con un selector entre las dos de `examples/session-12/` servidas por el backend) y
  `POST /agent-estimations` (ejecuta y renderiza el resultado).
- `views/agent-estimations/new.ejs` y `show.ejs` + parciales `_estimate.ejs` (informe: componentes
  con horas, fuentes y rationale; total; suposiciones; confianza) y `_steps.ejs` (timeline: un
  bloque por `AgentStepRecord` con razonamiento, acción y observación, y la cabecera/cierre de
  `trace_text`).
- Botón **Descargar traza (.txt)** que baja el `trace_text` tal cual (blob client-side, sin
  endpoint extra).

Con esto se cumple el requisito de trazabilidad del enunciado: trazas que se ven en la UI **y** un
`.txt` con la traza de ejecución.

### Paso 7 — Tests

Planos, con dobles, sin red ni Postgres. Ficheros nuevos:

- `tests/test_agent_tools.py`:
  - `calculate_estimate`: mediana + contingencia con varias referencias; componente sin
    referencias → `unbudgeted`, 0 h y total que no lo cuenta; determinismo (misma entrada, misma
    salida).
  - `validate_estimate`: componente sin referencias → issue; estimación fuera de rango → issue;
    total incoherente → issue; estimación limpia → `ok=True` sin issues.
  - `dispatch_tool`: tool desconocida lanza `ValueError`; argumentos inválidos disparan la
    `ValidationError` que el runner convierte en observación de error.
- `tests/test_agent_runner.py` — un doble del cliente con respuestas guionizadas:
  - Un turno con `function_call` y un turno final → 2 pasos de traza, `stopped_reason="completed"`,
    informe parseado.
  - El `function_call_output` reenviado lleva el `call_id` de su call.
  - Argumentos que no son JSON válido → observación de error, el bucle sigue.
  - Dos calls en el mismo turno → dos registros de traza, un único summary en el primero.
  - El doble siempre devuelve `function_call` → para por `max_iterations` con `report=None`.
- `tests/test_agent_endpoint.py` — con dependency overrides: sin clave/embedder → 500; con runner
  falso → 200 y cuerpo con `trace`, `trace_text` e `iterations` coherentes.

Y el contrato de capas ya existente (`tests/test_layering.py`) debe seguir en verde sin tocarlo:
esa es la prueba de que la inyección del backend está bien hecha.

### Paso 8 — Documento de resultados

`docs/sesion-12-capa-agentica.md`, con:

1. Qué se construyó (paquete, tools, bucle, endpoint, UI) y qué se reutilizó.
2. La **traza real** de la ejecución de entrega sobre `sample_transcript_complex.txt` (la misma que
   `examples/session-12/trace_complex.txt`): pasos con razonamiento + acción + observación,
   número de iteraciones y motivo de parada.
3. El informe final (componentes, horas, fuentes, suposiciones, confianza).
4. Una nota de 2–3 frases sobre el comportamiento del agente: cómo repartió las búsquedas por
   componente, cómo reaccionó a lo que devolvió cada tool y si el bucle paró solo a la primera.

## Entregables

- `app/generation/agentic/estimation_agent/` (`schemas.py`, `tools.py`, `runner.py`).
- `get_async_openai_client()` y `get_budget_search_backend()` en `app/dependencies.py`; settings
  del agente en `app/config.py`.
- `app/api/agent_estimation.py` con `POST /api/v1/agent/estimate/from-transcript` y su registro en
  `app/main.py`.
- `scripts/run_agent_s12.py`.
- `examples/session-12/`: las dos transcripciones y el esqueleto tomados de la referencia,
  `offline_budget_stub.py` propio y `trace_complex.txt` con la traza de la ejecución de entrega.
- UI en `cliente-web/` (ruta, cliente, vistas, descarga de traza).
- `tests/test_agent_tools.py`, `tests/test_agent_runner.py`, `tests/test_agent_endpoint.py`.
- `docs/sesion-12-capa-agentica.md` con la evidencia real.

## Criterios de aceptación

Con `sample_transcript_complex.txt`, `gpt-5` y effort `medium`:

- [ ] El agente identifica **más de un componente** y hace **más de una llamada** a
      `search_budgets` (una por componente). Si resuelve todo en una sola búsqueda, o el prompt es
      demasiado directivo o las descripciones de las tools no guían bien.
- [ ] Llama a `calculate_estimate` con los componentes y sus referencias recuperadas.
- [ ] Llama a `validate_estimate` como último paso y subsana (o reporta) lo que flaggee.
- [ ] Termina por sí solo: `stopped_reason="completed"`, sin cortarse a mitad y sin agotar
      `max_iterations`.
- [ ] Produce un informe estructurado coherente: total = suma de componentes, sin componentes
      `unbudgeted` silenciosos, suposiciones explícitas.
- [ ] La traza muestra, para cada paso, razonamiento + acción + observación, y la UI la pinta y la
      descarga como `.txt`.
- [ ] `uv run pytest`, `ruff` y `mypy` en verde, incluido `tests/test_layering.py` sin cambios.

## Checklist de verificación

- [ ] Los schemas de tools son planos (`type`/`name` al mismo nivel), `strict: true`,
      `additionalProperties: false` en todos los niveles y todas las propiedades en `required`.
- [ ] Cada `function_call_output` referencia el `call_id` de su `function_call`.
- [ ] El bucle tiene condición de parada natural **y** `max_iterations` como salvaguarda.
- [ ] Los argumentos se validan con Pydantic antes de ejecutar la tool; el fallo se devuelve al
      modelo, no rompe el bucle.
- [ ] `search_budgets` envuelve `retrieve()`; no hay ninguna búsqueda reimplementada en
      `estimation_agent/`.
- [ ] `estimation_agent/` no importa `app.generation.rag` ni `app.dependencies` (lo comprueba
      `test_layering.py`).
- [ ] El backend de búsqueda hereda el `RuntimeRetrievalConfig` vigente (lo que diga Ajustes).
- [ ] Todo el código, docstrings, descripciones de tools, prompts y logs están en inglés.
- [ ] El system prompt describe el método (descomponer → buscar por componente → calcular →
      validar → parar) y las tres tools disponibles.
- [ ] El endpoint devuelve 500 genérico sin clave o sin embedder, como el resto del repo.
- [ ] El script tiene documentada la disciplina de coste (mini+simple para depurar, gpt-5+medium
      para entregar).
- [ ] `docs/sesion-12-capa-agentica.md` contiene la traza real, no un ejemplo inventado.
- [ ] La rama es `feature/pre-session-12` y el PR apunta a `main`.

## Documentación de referencia

- Spec de la Sesión 11 (citación verificable y RAGAS):
  [`sesion-11-citacion-verificable-ragas.md`](./sesion-11-citacion-verificable-ragas.md)
- Spec de la Sesión 10 (búsqueda híbrida y reranking):
  [`sesion-10-busqueda-hibrida-reranking.md`](./sesion-10-busqueda-hibrida-reranking.md)
- Documento de la Sesión 10 multifuente:
  [`../docs/sesion-10-multifuente.md`](../docs/sesion-10-multifuente.md)
- Patrón actor–critic–boss del repo:
  [`../docs/patron-actor-critic-boss.md`](../docs/patron-actor-critic-boss.md)
- OpenAI — Responses API y function calling:
  https://platform.openai.com/docs/guides/function-calling
- OpenAI — reasoning (esfuerzo y summaries):
  https://platform.openai.com/docs/guides/reasoning
- Pydantic v2 — protocolos y validación:
  https://docs.pydantic.dev/latest/concepts/models/

## Nota

**Qué es nuestro (y por qué).** El repo de referencia resuelve la sesión con
`agent_loop`/`agent_schemas`/`agent_tools` planos bajo `generation/agentic/`, un script CLI y nada
más: ni endpoint ni UI ni layering. Aquí el paquete ya tenía `boss` y `critic`, el contrato de
capas de `test_layering.py` prohíbe que el agente importe el retrieval, y el enunciado pide que el
backend de negocio siga enviando transcripciones y que las trazas se vean. De ahí las decisiones
propias: paquete `estimation_agent` con el backend inyectado desde el composition root, endpoint
`/api/v1/agent/estimate/from-transcript`, UI de trazas y `validate_estimate` obligatoria. De la
referencia se toman solo las dos transcripciones y el esqueleto de cálculo: son material de
ejercicio, no arquitectura.

**Lo que NO hay que hacer.** No uses el modo agéntico interno de la Responses API para que la API
encadene sola: el bucle se conduce a mano o no hay traza que mostrar. No reimplementes la
recuperación dentro de la tool. No inventes horas para un componente sin referencias: se marca
`unbudgeted` y se devuelve el problema al modelo. No persigas perfección en los reasoning
summaries: con efforts baratos llegan vacíos y el paso queda honesto con
`(no reasoning summary emitted)`. Y no copies el `example_trace` de la referencia: la entrega es
tu propia ejecución.

**La trampa de la sesión.** Un agente que elige mal la tool o inventa argumentos casi nunca es
culpa del modelo: es la **descripción** de la tool, que es lo único que el modelo lee. Si el agente
intenta resolver un proyecto multicomponente con una única búsqueda gigante, no toques el bucle:
reescribe la descripción de `search_budgets` para que diga, en inglés claro, que se llama una vez
por componente y con una query enfocada. Y la segunda trampa, la silenciosa: olvidar el `call_id`
al devolver el resultado. La API no siempre falla ruidosamente; a veces el modelo simplemente se
queda sin la observación y da vueltas. El test del doble de cliente existe para que eso no llegue
a producción.
