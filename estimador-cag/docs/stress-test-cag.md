# Stress test del CAG

> Cómo medir los límites del CAG (latencia, coste y memoria) antes de introducir
> RAG, con una campaña reproducible que corre sin LLM en los tests.

## Qué es y para qué sirve

Hasta la sesión 5 el estimator es un sistema **CAG** (Cache-Augmented Generation):
cada turno inyecta en el prompt la memoria del proyecto (`summary` + `anchors` +
ventana deslizante + `ProjectMetadata` + tier), el turno actual y el texto de los
adjuntos. Todo cabe en el contexto *por construcción*, porque el sistema se pensó
para proyectos cortos y adjuntos modestos.

`evals/stress/` pone ese "por construcción" a prueba. Responde con datos a tres
preguntas que ningún test verde contesta:

- ¿A qué turno el sistema empieza a **olvidar** el nombre del proyecto?
- ¿Cuánto cuesta el turno 20 frente al turno 1?
- ¿A qué tamaño de adjunto la latencia **P95** supera el SLA del cliente?

Es el baseline cuantitativo del CAG sobre el que, en el directo, se compara RAG.

> **Fallo duro vs degradación silenciosa.** Un esquema que revienta (502,
> `ValidationError`) enciende el CI y te avisa. Una degradación que no rompe nada
> —el recall del proyecto cae del 90 % al 60 %, la latencia sube 200 ms por
> turno— no enciende ningún test rojo. En un CAG, la degradación silenciosa es la
> peligrosa: el sistema sigue respondiendo y el olvido solo se ve en producción.
> Este harness convierte esa degradación invisible en números.

## La observabilidad: `turn_observed`

Antes de medir hacía falta **una** fila por turno. Hasta la sesión 5 la estimación
emitía logs sueltos (`session.estimate_received`, `llm.structured.end`,
`history_compressed`, `summarizer_completed`); reconstruir un CSV exigía
reconciliar cinco líneas por timestamp. Ahora cada turno emite un único evento
agregado, definido en `app/schemas/observability.py` (`TurnObservation`):

| Campo | Qué mide |
| --- | --- |
| `turn_index` | Turno 1-based dentro de la sesión (contador monótono). |
| `session_id` | Sesión a la que pertenece el turno. |
| `enriched_transcript_chars` | Caracteres de transcript + adjuntos concatenados. |
| `attachments_total_chars` | Caracteres extraídos del adjunto (0 si no hay). |
| `messages_in_window` | Mensajes en la ventana deslizante tras la compresión. |
| `anchors_count` | Anclas acumuladas. |
| `summary_chars` | Caracteres del resumen acumulativo. |
| `tokens_in` / `tokens_out` | Consumo real de la llamada al LLM. |
| `cost_usd` | Coste estimado de la llamada. |
| `latency_ms` | Latencia de la llamada. |
| `cache_hit_kind` | `none` \| `exact` \| `semantic` (el path conversacional siempre `none`). |
| `last_resolved_tier` | Tier de audiencia resuelto. |

Se emite justo antes del `return` en
`EstimationService.estimate_conversational()` (`app/services/estimation.py`), que
es el único punto con acceso a la vez al `meta` del wrapper (tokens/coste/latencia)
y al `session.history` **tras** la compresión. Dos decisiones de esa emisión:

- **`turn_index` es un contador en la `Session`** (`session.turn_count`), no
  `len(history.messages)//2`: la compresión desaloja pares, así que la longitud del
  historial no sirve como índice.
- **`cache_hit_kind` existe para ambos caminos**, pero `estimate_conversational`
  nunca cachea: siempre emite `none`. Es un dato del baseline, no un bug.

El evento se emite a structlog (observabilidad en producción) y además se guarda
en `Session.last_observation`, que `GET /api/v1/sessions/{id}` expone como
`last_turn` junto al texto de `summary` y `anchors`. El runner lee de ahí, **no de
stdout**: es determinista y funciona igual en modo in-process que sobre HTTP.

## Estructura

```
evals/stress/
├── __init__.py                 # re-exports (escenarios + métricas)
├── scenarios.py                # perfiles multi-turno + fact-tracker
├── metrics.py                  # LatencyBudget / CostBudget / MemoryDrift
├── run.py                      # CLI de la campaña (python -m evals.stress.run)
├── REPORT.md                   # el deliverable: tabla + curvas + lectura
└── fixtures/
    └── build_pdfs.py           # PDFs sintéticos deterministas (versionados)
```

## Los tres escenarios

`evals/stress/scenarios.py` describe cada perfil como una lista inmutable de
`Turn(turn_index, transcript, fact_to_remember)` dentro de un `Scenario`. El
`fact_to_remember` es un string corto y **literal** que `MemoryDriftMetric`
buscará en snapshots posteriores; el `anchor_fact` es el hecho persistente cuya
supervivencia se mide turno a turno.

| Perfil | Qué fuerza | `anchor_fact` | Turno del ancla |
| --- | --- | --- | --- |
| `growing` | El proyecto crece con requisitos coherentes (SSO, multi-tenant, audit log, CSV…). | `Nimbus` | 1 |
| `pivot` | El turno 5 cambia el stack (React → Flutter): ¿se actualiza la metadata o se acumula? | `Flutter` | 5 |
| `contradiction` | Presupuesto 30k (t3) contradicho por 80k (t8): ¿qué valor se preserva? | `30000 EUR` | 3 |

## Fixtures de adjuntos

`evals/stress/fixtures/build_pdfs.py` genera PDFs de **5, 20, 50 y 100 KB** de
texto plano (el `0 KB` es ausencia de adjunto). Cada PDF lleva en la primera línea
un marcador estable (`NIMBUS-ATTACH-MARKER-<kb>KB`) para que el recall del
contenido sea medible por match literal.

- **Determinismo**: fpdf2 estampa fecha de creación por defecto; la fijamos y
  calibramos el relleno con una búsqueda binaria. Dos ejecuciones producen bytes
  idénticos.
- **Versionados**: los PDFs generados se commitean en `evals/stress/fixtures/`
  junto a `results.csv` como evidencia del baseline; el runner los regenera si
  faltan.
- La utilidad es el **tamaño calibrado**: a partir de ~20 KB el texto extraído
  roza el tope `MAX_ATTACHMENT_CHARS=60.000`, así que el truncado se ve en acción.

## Las métricas

`evals/stress/metrics.py` añade tres métricas binarias que reutilizan el tipo
`MetricResult(name, score, passed, details)` del framework existente
(`evals/metrics.py`), pero **viven en `evals/stress/`**: operan sobre
`turn_observed` y el snapshot de sesión, no sobre `(GoldenCase, EstimationResult)`.
Meterlas en `evals/metrics.py` obligaría a aflojar la firma de `run_all_metrics` y
mezclaría dos dominios.

- **`LatencyBudgetMetric(budget_ms)`** — 1.0 si `latency_ms <= budget_ms`. Convierte
  el SLA en un test: el presupuesto es un contrato de diseño, no una bandera a
  observar a posteriori.
- **`CostBudgetMetric(budget_usd)`** — 1.0 si `cost_usd <= budget_usd`.
- **`MemoryDriftMetric(fact, where=("summary","anchors","metadata"))`** — 1.0 si el
  hecho aparece (case-insensitive) en alguno de esos campos del snapshot.

Dos reglas de diseño: **nada de embeddings ni LLM-as-judge** (determinismo >
sofisticación) y **match exacto case-insensitive** sobre el texto del snapshot. Un
hecho que no aparece literalmente cuenta como drift: eso es la señal.

## El runner

`evals/stress/run.py` orquesta `escenario × tamaño de adjunto × repetición ×
turno` y vuelca un CSV con **una fila por turno**: los 13 campos de
`turn_observed`, las tres métricas y tres columnas de contexto (`scenario`,
`attachment_kb`, `repeat`).

- Reusa el patrón de `evals/run.py`: `TestClient` en proceso por defecto, `httpx`
  con `--http BASE`.
- El CSV se escribe **fila a fila** (cabecera + `flush`), de modo que una campaña
  larga se puede seguir en vivo y no se pierde lo medido si se interrumpe.
- Cada celda abre una sesión nueva; el adjunto del tamaño elegido se envía en
  cada turno del escenario.

## Cómo leer el reporte

`REPORT.md` contiene la **tabla resumen** (P50/P95 de latencia, coste acumulado,
hit rate de ambas cachés y recall medio del fact-tracker), **tres curvas** en
formato tabla y **dos párrafos de lectura**:

1. latencia vs tokens de entrada;
2. coste acumulado vs turno;
3. `MemoryDriftMetric` vs N.

En la campaña de referencia (`deepseek-v4-flash`, 1 repetición) los titulares son:

- **La latencia rompe primero**: 0/33 turnos cumplen el SLA de 4 s (P95 55–62 s).
- **El coste por turno crece ×5,2** del turno 1 al 10 en `growing` (tokens ×9,4).
- **Cachés 0 %**: el path conversacional no cachea.
- **El `project_name` sobrevive** (recall 1,00) vía `ProjectMetadata`, pero el
  hecho de presupuesto da 0,00 por un fallo silencioso (formato/anclas en inglés).
- **Los adjuntos ≥ 20 KB se truncan** a 60.000 caracteres: 20, 50 y 100 KB aportan
  el mismo contexto y aun así cuestan distinto.

Detalle completo, tablas y lectura en [REPORT.md](../evals/stress/REPORT.md).

## Ficheros y configuración

| Pieza | Fichero |
| --- | --- |
| Observación del turno | `app/schemas/observability.py` |
| Emisión + persistencia | `app/services/estimation.py` (`estimate_conversational`) |
| Exposición en debug | `app/routers/sessions.py` (`GET /sessions/{id}`) |
| Escenarios | `evals/stress/scenarios.py` |
| Fixtures PDF | `evals/stress/fixtures/build_pdfs.py` |
| Métricas | `evals/stress/metrics.py` |
| Runner | `evals/stress/run.py` |
| Reporte | `evals/stress/REPORT.md` |

Knobs del runner: `--latency-budget-ms` (SLA, por defecto 4000),
`--cost-budget-usd` (0.05), `--growing-turns` (20), `--repeats` (3) y
`--attachment-sizes` (`0,5,20,50,100`). La generación de PDFs requiere `fpdf2`
(dependencia **dev**, no de runtime).

## Cómo ejecutarlo

```bash
# Smoke: rápido, in-process (TestClient), proveedor real
uv run python -m evals.stress.run \
  --scenarios growing --attachment-sizes 0 --repeats 1 --growing-turns 2 \
  --output /tmp/smoke.csv

# Contra un servidor ya arrancado
uv run python -m evals.stress.run --http http://localhost:8000 \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,5,20,50,100 \
  --repeats 3 \
  --output evals/stress/results.csv
```

La campaña completa llama al proveedor de verdad (~1 llamada de estimación + 1 de
metadata por turno) y, con un modelo lento, puede tardar horas. Empieza por el
smoke.

## Cómo ejecutarlo con Docker

El runner vive en `evals/` y usa `fpdf2` (dependencia **dev**): la imagen
`runtime` (servicio `api`) **no** los incluye. Por eso el runner corre en la
imagen `test`, y `api` solo se usa en el modo `--http`. `.env` está en
`.dockerignore`, pero Compose lo inyecta vía `env_file`, así que las claves llegan
a los contenedores.

```bash
cd estimador-cag

# 1. Imagen test (dev deps + evals)
docker compose --profile test build test

# 2. Tests nuevos sin LLM
docker compose --profile test run --rm test \
  uv run --no-sync pytest tests/test_stress_metrics.py tests/test_stress_runner.py -q

# 3a. Campaña in-process (TestClient, no necesita api).
#     El volumen saca el CSV del contenedor efímero.
docker compose --profile test run --rm \
  -v "$PWD/evals/stress:/app/evals/stress" \
  test uv run --no-sync python -m evals.stress.run \
    --scenarios growing --attachment-sizes 0 --repeats 1 --growing-turns 2 \
    --output evals/stress/results.csv

# 3b. Campaña contra api por HTTP (api se resuelve por nombre de servicio)
docker compose up -d api
docker compose --profile test run --rm \
  -v "$PWD/evals/stress:/app/evals/stress" \
  test uv run --no-sync python -m evals.stress.run \
    --http http://api:8000 \
    --scenarios growing,pivot,contradiction \
    --attachment-sizes 0,5,20,50,100 --repeats 1 \
    --output evals/stress/results.csv
```

Comprobar el endpoint de debug y el evento `turn_observed` en la API dockerizada:

```bash
docker compose up -d api
SID=$(curl -s -X POST localhost:8000/api/v1/sessions | jq -r .session_id)
curl -s -X POST "localhost:8000/api/v1/sessions/$SID/estimate" \
  -F transcript="Queremos un CRM llamado Nimbus." \
  -F project_type=web_saas -F detail_level=medium -F output_format=phases_table \
  | jq '.result.summary'
curl -s "localhost:8000/api/v1/sessions/$SID" | jq .last_turn
docker compose logs -f api | grep turn_observed
```

Notas:

- **No** ejecutes el runner en `docker compose run api …`: esa imagen no trae
  `evals/` ni `fpdf2` (fallará con `ModuleNotFoundError`).
- El path conversacional no usa caché, así que `CACHE_BACKEND=redis` no bloquea la
  campaña aunque Redis no esté sano.
- Sin el `-v` del `run`, `results.csv` se queda dentro del contenedor y se pierde
  con `--rm`.
- Cierra con `docker compose down` (añade `-v` para borrar el volumen de Postgres).

## Cómo se prueba

- `tests/test_stress_metrics.py` — las tres métricas con casos que pasan, fallan y
  límite (sin red).
- `tests/test_stress_runner.py` — la campaña end-to-end con un wrapper LLM falso:
  comprueba que el CSV sale con todas las columnas, una fila por turno y las
  métricas evaluadas.
- `tests/test_sessions_endpoints.py` — que `GET /sessions/{id}` expone `last_turn`
  con los campos correctos y cuenta los caracteres del adjunto.

## Decisiones de diseño y límites

- **Evento agregado en vez de cinco logs**: parseo a CSV de una pasada, correlación
  trivial entre `messages_in_window` y `cost_usd`, sin reconciliar timestamps.
- **Snapshot en vez de parsear stdout**: el runner lee `last_turn` de
  `GET /sessions/{id}`; es determinista y vale igual in-process que por HTTP, sin
  acoplarse al renderer de structlog.
- **Métricas en `evals/stress/`**: dependen del shape de `turn_observed`/snapshot;
  se reutiliza el *tipo* `MetricResult`, no la firma del golden dataset.
- **Fixtures y `results.csv` versionados**: son deterministas y valen como
  evidencia del baseline; el runner los regenera (sobrescribe) al repetir la
  campaña.
- **Un único proveedor**: las curvas solo son comparables si el modelo no cambia.

Límites conocidos de la campaña de referencia: `--repeats 1` (P95 sobre 8–10
muestras), un único modelo con *reasoning* de latencia alta, y `MemoryDriftMetric`
por **match exacto**, sensible al formato (`30.000` vs `30000 EUR`) y a que las
anclas heurísticas solo reconocen frases en inglés. Por eso el ejercicio **mide, no
optimiza**: mover una constante (acortar el system, bajar `MAX_CONVERSATION_TURNS`)
invalidaría la comparación contra RAG.
