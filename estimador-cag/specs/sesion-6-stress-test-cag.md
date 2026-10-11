# Spec — Proyecto 1: Stress test del CAG (baseline de latencia, coste y memoria)

## Objetivo

Hasta la sesión 05 el estimator es un sistema **CAG** (Cache-Augmented Generation): cada turno inyecta en el prompt la memoria del proyecto (`summary` + `anchors` + ventana deslizante + `ProjectMetadata` + tier), el turno actual y el texto extraído de los adjuntos. Todo cabe en el contexto del LLM **por construcción**, porque el sistema se diseñó para proyectos cortos y adjuntos modestos.

Ese "por construcción" nunca se ha puesto a prueba. No sabemos:

- A qué turno el sistema empieza a **olvidar** el nombre del proyecto.
- Cuánto cuesta el turno 20 frente al turno 1.
- A qué tamaño de adjunto la latencia **P95** supera el SLA del cliente.

La sesión 06 (directo) introduce **RAG** como respuesta a esas limitaciones. Pero antes de aceptar la solución, el alumno tiene que **ver con sus propios datos qué limitación existe**. Este ejercicio es ese trabajo: instrumentar el CAG, someterlo a tres escenarios de carga (multi-turno largo, adjuntos grandes, ráfagas) y producir un `REPORT.md` con tres curvas y dos párrafos de lectura.

Al terminar este ejercicio tendrás:

- Un evento estructurado `turn_observed` por cada llamada a `POST /api/v1/sessions/{id}/estimate`, con los 13 campos que resumen el turno.
- Tres perfiles sintéticos de conversación (`growing`, `pivot`, `contradiction`) con un `fact-tracker` que declara, turno a turno, qué hecho debe sobrevivir.
- Un corpus determinista de PDFs sintéticos de tamaños calibrados (5/20/50/100 KB) que el runner regenera.
- Tres métricas binarias nuevas (`LatencyBudgetMetric`, `CostBudgetMetric`, `MemoryDriftMetric`) sobre el patrón `MetricResult` existente.
- Un runner CLI `evals/stress/run.py` que vuelca `evals/stress/results.csv` con una fila por turno.
- El deliverable: **`evals/stress/REPORT.md`**, un baseline cuantitativo del CAG sobre el que comparar RAG en el directo.

> **Reconciliación con el repo.** El punto de partida es `main` con la sesión 05 cerrada: sesiones en memoria, ventana deslizante, anclas heurísticas, resumen acumulativo, `ProjectMetadata` extraído turno a turno, tier dinámico, Actor-Critic-Boss opcional, adjuntos PDF/DOCX por Camino B, framework de evals con 16 casos golden y wrapper LLM observable. **Nada se reescribe**: la instrumentación es aditiva y todo lo demás se apoya en lo que ya existe.

## Contexto del proyecto

Piezas relevantes que se reutilizan o extienden (todas ya en `main`):

- `app/services/estimation.py` — `EstimationService.estimate_conversational()` es el único punto donde se puede agregar la observación del turno: tiene a la vez el `meta` del wrapper (tokens, coste, latencia) y el `session.history` (ventana, anclas, resumen) tras la compresión. Aquí nace `turn_observed`.
- `app/sessions/models.py` — `Session`, `ConversationHistory` (con `messages`, `anchors`, `summary`), `ProjectMetadata`. La observación del último turno se guarda en la propia `Session` para poder exponerla.
- `app/sessions/compression/` — `AnchorDetector`, `CumulativeSummarizer`, `CompressionPolicy`. Fuente de `anchors_count` y `summary_chars`.
- `app/sessions/tier_resolver.py` — `resolve_tier()` devuelve `(Tier, regla)`. Fuente de `last_resolved_tier`.
- `app/attachments/extractor.py` — `extract_text()` / `enrich_transcript()`. El router ya conoce los caracteres extraídos antes de llamar al servicio.
- `app/services/llm_wrapper.py` — cada llamada estructurada devuelve `meta` con `model`, `provider`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`, `fallback_used`; tabla de precios en `MODEL_COSTS`.
- `app/routers/sessions.py` — `GET /api/v1/sessions/{id}` ya expone `message_count`, `anchors_count`, `summary_chars`, `last_resolved_tier`, `last_tier_rule`: extendemos este endpoint con el último `turn_observed` en lugar de parsear stdout.
- `evals/metrics.py` — `MetricResult(name, score, passed, details)` y `run_all_metrics`; el patrón de métrica binaria ya está.
- `evals/run.py` — runner CLI con dos transportes: `TestClient` en proceso (por defecto) y `httpx` con `--http`. El runner de estrés reusa ambos patrones.

## Requisitos para el ejercicio

- Proyecto de la sesión 05 en verde (`uv run pytest`, `uv run ruff check .`, `uv run mypy app frontend`).
- Dependencia nueva **solo para generar fixtures** (dev): `fpdf2`. No entra en runtime.
- Los PDFs sintéticos son deterministas; este repo **versiona** los generados en `evals/stress/fixtures/` como evidencia del baseline. El runner los regenera si faltan.
- API key configurada para poder medir de verdad. Los tests de métricas `tests/test_stress_metrics.py` no tocan el LLM.
- Un **único proveedor/modelo** durante toda la campaña: las curvas solo son comparables si el modelo no cambia.

```bash
uv add --dev fpdf2
```

Los artefactos de la campaña (`evals/stress/fixtures/*.pdf` y `evals/stress/results.csv`) se versionan; se regeneran con el runner cuando se repite la campaña.

## ✍ Ejercicio

Cinco bloques con dificultad escalonada. Los cinco son medibles: el deliverable es un reporte con números, no código de producción.

### Paso 1 — Emitir el evento `turn_observed`

Hoy el sistema emite logs sueltos a lo largo de cada estimación: `session.estimate_received`, `llm.structured.end`, `history_compressed`, `summarizer_completed`, `tier_resolved`. Para extraer un CSV de una pasada necesitas que cada llamada deje **un único evento** con todo lo relevante junto. Un evento agregado es más útil que cinco logs separados porque: (a) el parseo a CSV es de una pasada, (b) la correlación entre `messages_in_window` y `cost_usd` es trivial (misma fila), y (c) no hay que reconciliar timestamps ni `call_id` entre líneas.

**Contrato de la observación** (`app/schemas/observability.py`, nuevo). Un `BaseModel` para que la `Session` pueda guardarlo y `GET /sessions/{id}` lo serialice sin conversión:

```python
class TurnObservation(BaseModel):
    """Foto agregada de un turno conversacional (bloque 1)."""

    turn_index: int  # 1-based, contado dentro de la sesión
    session_id: str
    enriched_transcript_chars: int  # transcript + adjuntos concatenados
    attachments_total_chars: int  # 0 si no hay adjuntos
    messages_in_window: int  # len(history.messages) tras compresión
    anchors_count: int
    summary_chars: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    cache_hit_kind: Literal["none", "exact", "semantic"] = "none"
    last_resolved_tier: str | None = None

    def to_event(self) -> dict[str, Any]:
        """Payload plano para structlog (una sola línea agregada)."""
        return self.model_dump()
```

**Emisión en el servicio.** En `EstimationService.estimate_conversational()` (justo antes del `return`):

```python
def estimate_conversational(
    self,
    *,
    session,
    transcript,
    project_type,
    detail_level,
    output_format,
    tier=None,
    version=None,
    attachment_chars: int = 0,  # NUEVO — el router ya lo conoce
) -> SessionEstimation: ...
```

```python
# ... tras apply_compression(...) y update_metadata(...)

observation = TurnObservation(
    turn_index=len(session.history.messages) // 2,  # (ver nota)
    session_id=session.session_id,
    enriched_transcript_chars=len(transcript),
    attachments_total_chars=attachment_chars,
    messages_in_window=len(session.history.messages),
    anchors_count=len(session.history.anchors),
    summary_chars=len(session.history.summary or ""),
    tokens_in=meta.get("input_tokens"),
    tokens_out=meta.get("output_tokens"),
    cost_usd=meta.get("cost_usd"),
    latency_ms=meta.get("latency_ms"),
    cache_hit_kind="none",  # el path conversacional no cachea
    last_resolved_tier=resolved_tier.value,
)
session.last_observation = observation
logger.info("turn_observed", **observation.to_event())
```

> **`turn_index`.** Calcúlalo al **principio** del método como `len(session.history.messages) // 2 + 1` (antes del `append`) y pásalo a la observación. Contar al final (`// 2`, tras el append) da el mismo número, pero es frágil si en el futuro la compresión desaloja dentro del propio turno. Elige uno y documéntalo.

> **`cache_hit_kind`.** El contrato admite `"exact"` y `"semantic"` para que el tipo sirva también al path transaccional (donde sí hay cachés), pero `estimate_conversational` **nunca cachea**, así que el valor emitido es siempre `"none"`. Esto es información del baseline: en multi-turno las cachés no ayudan.

> **`attachment_chars`.** Los caracteres del adjunto se conocen en el router (`sum(len(t) for _, t in extracted)`), no en el servicio. El servicio no vuelve a contar nada: recibe el entero. Evita re-parsear la transcripción enriquecida para adivinar dónde acaba el transcript y empieza el adjunto.

**Transporte al runner.** Extiende `GET /api/v1/sessions/{id}` para devolver el último turno observado, en lugar de parsear stdout (frágil y dependiente del formato de logging):

```python
class SessionInfoResponse(BaseModel):
    # ... campos ya existentes ...
    last_turn: TurnObservation | None = None
```

### Paso 2 — Diseñar los tres perfiles multi-turno

Crea `evals/stress/scenarios.py`. Cada perfil es una lista de tuplas `(turn_index, transcript, fact_to_remember)`, y el `fact_to_remember` es el hecho que **MemoryDriftMetric** buscará en snapshots de turnos posteriores. Los facts son **strings cortos y literales** de match exacto (case-insensitive) sobre el texto del snapshot: `"Nimbus"`, `"Flutter"`, `"30000 EUR"`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    turn_index: int
    transcript: str
    fact_to_remember: str | None = None  # None = turno sin hecho durable


@dataclass(frozen=True)
class Scenario:
    name: str  # "growing" | "pivot" | "contradiction"
    turns: tuple[Turn, ...]
    project_type: str = "web_saas"
    detail_level: str = "medium"
    output_format: str = "phases_table"


def build_growing(n: int = 20) -> Scenario: ...
def build_pivot() -> Scenario: ...
def build_contradiction() -> Scenario: ...
def all_scenarios() -> tuple[Scenario, ...]: ...
```

Fragmento de cada perfil (los tres diseñados para forzar un comportamiento concreto):

```python
# growing — turno a turno se añaden requisitos coherentes. El project_name
# ("Nimbus") se declara en el turno 1 y debe sobrevivir hasta el 20.
(Turn(1, "Queremos estimar un CRM llamado Nimbus para el equipo de ventas.", "Nimbus"),)
(Turn(2, "Súmale autenticación con SSO y roles por tenant.", "SSO"),)
(Turn(3, "Necesita multi-tenant con aislamiento de datos.", "multi-tenant"),)
(Turn(4, "Añade un audit log de todas las acciones.", "audit log"),)
# ... hasta el turno 20 (exporte CSV, integración de facturación, informes…)

# pivot — el turno 5 cambia el stack (React -> Flutter). El pivote debe
# actualizar la metadata sin acumular ambos en mentioned_technologies.
(Turn(1, "App de campo en React para inspectores.", "React"),)
(Turn(5, "Cambiamos de idea: lo haremos en Flutter.", "Flutter"),)

# contradiction — presupuesto 30k en el turno 3, 80k en el turno 8. Mide qué
# valor se preserva, cuál se promueve a ancla y cuál acaba en el summary.
(Turn(3, "El presupuesto está locked at 30000 EUR.", "30000 EUR"),)
(Turn(8, "Lo hemos ampliado: el presupuesto está locked at 80000 EUR.", "80000 EUR"),)
```

> **Por qué tuplas inmutables.** El runner consume el mismo `Scenario` en cada repetición; que sea inmutable evita que una mutación accidental contamine las 3 pasadas. El `fact-tracker` vive en los propios `Turn` (`fact_to_remember`), no en una estructura paralela: un solo sitio que mantener.

### Paso 3 — Generar el corpus de adjuntos sintéticos

`evals/stress/fixtures/build_pdfs.py`, con `fpdf2`. Cinco puntos de calibración: `0 KB` es ausencia de adjunto (baseline, no fichero), y `5/20/50/100 KB` son PDFs reales. El runner los regenera si faltan y este repo versiona los generados.

```python
SIZES_KB: tuple[int, ...] = (5, 20, 50, 100)


def build_pdf(*, path: Path, target_kb: int, seed: int = 0) -> Path:
    """Genera un PDF determinista de ~target_kb KB de texto plano."""


def build_all(out_dir: Path) -> dict[int, Path]:
    """Genera attach_5kb.pdf … attach_100kb.pdf y devuelve {kb: ruta}."""
```

Cada PDF lleva un **marcador estable y localizable** (por ejemplo `NIMBUS-ATTACH-MARKER-<kb>KB` en la primera línea) para que el recall del contenido del adjunto sea medible por match exacto. Puede usar Lorem Ipsum repetido como relleno: lo que importa es el **tamaño calibrado**, no la semántica.

> **Calibración.** `fpdf2` no garantiza el tamaño de salida: cuenta caracteres y ajusta el relleno hasta acercarse a `target_kb` (o acepta un rango por tamaño y documéntalo en el reporte). El punto didáctico es que el `100 KB` de texto plano ronda `MAX_ATTACHMENT_CHARS=60.000`, así que verás el truncado en acción.

### Paso 4 — Añadir las tres métricas

Crea **`evals/stress/metrics.py`** (no `evals/metrics.py`). Justificación: las tres métricas dependen del *shape* de `turn_observed` y del *snapshot* de la sesión, no de `(GoldenCase, EstimationResult)` como las métricas del golden dataset. Meterlas en `evals/metrics.py` obligaría a aflojar la firma de `run_all_metrics` y mezclaría dos dominios. Reutilizamos el **tipo** `MetricResult`, que sí es común.

```python
from evals.metrics import MetricResult


class LatencyBudgetMetric:
    """1.0 si latency_ms <= budget_ms; 0.0 si no. Convierte el SLA en un test."""
    name = "latency_budget"

    def __init__(self, budget_ms: int) -> None: ...
    def evaluate(self, observation) -> MetricResult: ...


class CostBudgetMetric:
    """1.0 si cost_usd <= budget_usd; 0.0 si no."""
    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None: ...
    def evaluate(self, observation) -> MetricResult: ...


class MemoryDriftMetric:
    """1.0 si el fact del turno k aparece en summary, anchors o ProjectMetadata
    del turno N (N > k); 0.0 si no. Match exacto case-insensitive."""
    name = "memory_drift"

    def __init__(self, fact: str, where: list[str] | None = None) -> ...
    def evaluate(self, session_snapshot) -> MetricResult: ...
```

Reglas de implementación:

- `LatencyBudgetMetric.evaluate(observation)` lee `observation.latency_ms`; si es `None` (llamada sin métricas) cuenta como fallo y lo deja en `details`.
- `CostBudgetMetric.evaluate(observation)` lee `observation.cost_usd`; `None` cuenta como fallo.
- `MemoryDriftMetric.evaluate(session_snapshot)` recibe el payload de `GET /sessions/{id}` (o un dict equivalente) y busca `fact.lower()` en: `summary` (substring), los `content` de `anchors` (substring), y los campos textuales de `metadata` (`project_name`, `agreed_scope`, `mentioned_technologies` join) — según `where`.
- **Determinismo > sofisticación**: ni embeddings ni LLM-as-judge. Match exacto case-insensitive. Un fact que no aparece literalmente no cuenta: eso es precisamente la señal de drift.

Tests unitarios mínimos en `tests/test_stress_metrics.py`: una métrica que pasa, una que falla, y un caso límite (p. ej. `latency_ms == budget_ms` pasa; `fact` presente solo en `anchors` y `where=["summary"]` falla).

### Paso 5 — Implementar el runner

`evals/stress/run.py`. Reusa el patrón de `evals/run.py`: `TestClient` en proceso por defecto, `httpx` cuando se pasa `--http`.

```bash
uv run python -m evals.stress.run --http http://localhost:8000 \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,5,20,50,100 \
  --repeats 3 \
  --output evals/stress/results.csv
```

Esqueleto:

```python
async def main() -> int:
    args = _parse_args()
    _ensure_fixtures(args.attachment_sizes)  # build_pdfs.build_all
    scenarios = _select_scenarios(args.scenarios)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for size_kb in args.attachment_sizes:
            for repeat in range(args.repeats):
                session_id = await client.create_session()
                for turn in scenario.turns:
                    response = await client.estimate(
                        session_id,
                        transcript=turn.transcript,
                        project_type=scenario.project_type,
                        detail_level=scenario.detail_level,
                        output_format=scenario.output_format,
                        attachment=_attachment_for(size_kb),  # None si 0
                    )
                    snapshot = await client.get_session(session_id)  # trae last_turn
                    observation = snapshot["last_turn"]
                    metrics = [
                        LatencyBudgetMetric(args.latency_budget_ms).evaluate(observation),
                        CostBudgetMetric(args.cost_budget_usd).evaluate(observation),
                        MemoryDriftMetric(turn.fact_to_remember).evaluate(snapshot)
                        if turn.fact_to_remember
                        else None,
                    ]
                    rows.append(_row(scenario.name, size_kb, repeat, observation, metrics))
    _write_csv(args.output, rows)
```

Decisiones:

- **La observación viene de `GET /sessions/{id}` (`last_turn`)**, no de parsear stdout. Es determinista, funciona igual en modo in-process que `--http`, y no acopla el runner al renderer de structlog. El log `turn_observed` sigue emitiéndose para observabilidad en producción.
- **El runner añade tres columnas de contexto** (`scenario`, `attachment_kb`, `repeat`) y **una columna por métrica** (`latency_budget`, `cost_budget`, `memory_drift`) al CSV.
- **El adjunto del tamaño elegido se envía en cada turno del escenario**, de modo que la matriz escenario × tamaño mide la interacción completa. La curva pura de adjuntos se lee de las filas del turno 1 (misma estimación inicial, mismo `project_type`).
- `--repeats` importa porque la latencia es ruidosa: sin repeticiones no hay P95 creíble. `results.csv` (no JSON) es el formato acordado.

### Paso 6 — Escribir `REPORT.md`

`evals/stress/REPORT.md`, Markdown corto (1-2 páginas), generado a mano a partir del CSV. Estructura mínima:

1. **Tabla resumen** — una fila por escenario × tamaño de adjunto, con: `P50 latency_ms`, `P95 latency_ms`, `total cost USD`, `exact cache hit rate`, `semantic cache hit rate`, `mean recall del fact-tracker`.
2. **Tres curvas** (como tablas, sin gráficos):
   - `latency_ms` vs `tokens_in`.
   - `cost_usd` acumulado vs `turn_index` (por escenario).
   - `MemoryDriftMetric` vs `N` (turnos).
3. **Dos párrafos de lectura**: dónde empieza a romperse el CAG y por qué; qué dimensión domina la degradación (latencia, coste o pérdida de memoria); qué caso límite justifica saltar a RAG.

El reporte debe contener **al menos una afirmación cuantitativa concreta**, del estilo:

- "A partir del turno N=12, el recall del `project_name` cae por debajo del 60%."
- "El coste del turno 20 multiplica por X.X el del turno 1."

## Criterios de "hecho"

El ejercicio está completo cuando:

- [ ] Cada llamada a `POST /api/v1/sessions/{id}/estimate` emite un evento `turn_observed` con los 13 campos del bloque 1, y `GET /api/v1/sessions/{id}` devuelve el `last_turn`.
- [ ] `uv run python -m evals.stress.run --http http://localhost:8000` corre end-to-end sin errores y deja `evals/stress/results.csv` con ≥ 50 filas (3 escenarios × 5 tamaños × ≥ 3 repeticiones × N turnos).
- [ ] `evals/stress/metrics.py` expone `LatencyBudgetMetric`, `CostBudgetMetric`, `MemoryDriftMetric`, con tests unitarios verdes.
- [ ] `evals/stress/REPORT.md` contiene la tabla resumen, las tres curvas (en tabla) y los dos párrafos de lectura, con al menos una afirmación cuantitativa concreta.
- [ ] Todo está en el repo, **incluidos** los fixtures PDF (`evals/stress/fixtures/*.pdf`) y `evals/stress/results.csv`. El reporte es el deliverable.

## Checklist de verificación

- [ ] `app/schemas/observability.py` define `TurnObservation` con exactamente los 13 campos del bloque 1.
- [ ] `estimate_conversational` acepta `attachment_chars` y emite `turn_observed` una sola vez por turno, antes del `return`.
- [ ] `Session` guarda `last_observation` y `GET /api/v1/sessions/{id}` lo expone como `last_turn`.
- [ ] `turn_index` es 1-based y consistente entre turnos (documentado).
- [ ] `evals/stress/scenarios.py` define los tres perfiles (`growing`, `pivot`, `contradiction`) con `fact_to_remember` por turno.
- [ ] `evals/stress/fixtures/build_pdfs.py` genera PDFs deterministas de 5/20/50/100 KB con marcador localizable.
- [ ] `evals/stress/metrics.py` reutiliza `MetricResult` y hace match exacto case-insensitive (sin embeddings ni LLM).
- [ ] `tests/test_stress_metrics.py` cubre una métrica que pasa, una que falla y un caso límite.
- [ ] `evals/stress/run.py` soporta `--http` / in-process y vuelca el CSV con las columnas acordadas.
- [ ] `evals/stress/fixtures/*.pdf` y `evals/stress/results.csv` están versionados (`.gitignore` no los excluye).
- [ ] `uv run pytest`, `uv run ruff check .` y `uv run mypy app frontend` pasan.

## Documentación de referencia

- fpdf2 — API de texto y páginas: https://py-pdf.github.io/fpdf2/
- httpx — cliente asíncrono (`AsyncClient`, `ASGITransport`): https://www.python-httpx.org/async/
- FastAPI `TestClient` — pruebas en proceso: https://fastapi.tiangolo.com/tutorial/testing/
- structlog — logging estructurado (`bind`, eventos con kwargs): https://www.structlog.org/en/stable/
- pypdf — extracción de texto (para entender el tamaño efectivo de los adjuntos): https://pypdf.readthedocs.io/en/stable/user/extract-text.html

## Nota

**Fallo duro vs degradación silenciosa.** Un esquema que revienta (`ValidationError`, 502) detiene el CI y te avisa. Una degradación que no rompe nada —el recall del proyecto cae del 90% al 60%, la latencia sube 200 ms por turno— no enciende ningún test rojo. En un CAG, **la degradación silenciosa es más peligrosa**: el sistema sigue respondiendo, el usuario no ve el error, y el olvido solo se detecta cuando el cliente recibe una estimación sobre el proyecto equivocado. Este ejercicio existe para convertir esa degradación invisible en números.

**Los presupuestos son contratos, no banderas.** Un `token budget`, un `latency budget` y un `cost budget` no son cosas que se observan a posteriori: son límites que se declaran antes y que el sistema debe respetar. `LatencyBudgetMetric(budget_ms=4000)` convierte el SLA del cliente en un test booleano que pasa o falla en cada turno. Si el presupuesto se incumple, el diseño está mal, no la métrica.

**Las tres curvas canónicas.** Todo sistema basado en contexto muestra las mismas tres curvas al escalar: latencia frente a tokens (el prompt crece, la generación se ralentiza), coste acumulado frente a turnos (cada turno reenvía más historia), y recall de hechos frente a la longitud del historial (lo que se sale de la ventana se olvida salvo que el resumen o las anclas lo rescaten). Saber leerlas es saber dónde está la frontera del CAG.

**Cuándo "el contexto está lleno" no es un error del LLM.** Cuando el contexto se satura, el modelo no falla: degrada. Ese punto —el turno en el que el `project_name` deja de sobrevivir, el tamaño de adjunto en el que el truncado borra información— no es un bug del modelo, es una **decisión arquitectónica**: el momento exacto en el que CAG empieza a perder frente a RAG. Este baseline cuantifica esa frontera antes de cruzar el puente.

**Fuera de alcance.** Implementar RAG (es el directo), optimizar el CAG (acortar el system, comprimir más agresivo, bajar `MAX_CONVERSATION_TURNS`), comparar proveedores, notebooks con matplotlib, persistir a Postgres/SQLite, LLM-as-judge y UI nueva. El ejercicio **mide, no optimiza**: mover una constante invalida la comparación contra RAG.
