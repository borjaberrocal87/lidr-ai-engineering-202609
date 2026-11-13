# Spec — Proyecto 5: Citación verificable a nivel de línea y baseline RAGAS (Sesión 11)

## Objetivo

La Sesión 10 dejó dos caminos de estimación conviviendo:

1. El pipeline `from-transcript` de la Sesión 09: brief → retrieval → contexto → generación
   estructurada con horas en engineer-days y citas globales (`sources: list[Citation]`).
2. El wizard de la Sesión 10: estructura módulo → tarea **sin horas** y, después, un consenso de
   horas por tarea calculado con búsqueda vectorial sobre el corpus `historical_task`.

En ambos caminos la evidencia existe, pero está **en el sitio equivocado**. Las citas globales no
dicen qué presupuesto histórico respalda cada línea; el consenso de horas devuelve vecinos, pero no
una prueba verificable de que la cifra salió de un chunk concreto. Un lector —o un evaluador— no
puede responder a la pregunta que importa: *¿de qué presupuesto histórico se derivó esta línea?*

Esta sesión cierra ese hueco con dos piezas y una medición:

1. **Citación verificable por línea.** Cada tarea de la estimación transporta sus fuentes
   (`chunk_id`, presupuesto padre y `evidence` verbatim). Un validador posterior comprueba que cada
   `chunk_id` citado estaba realmente en el contexto recuperado. Una cita que apunta a un chunk que
   el LLM nunca vio es una **citación colgante**: un fallo de calidad, no un detalle cosmético.
2. **Detección de citaciones colgantes** después de generar, con informe estructurado, reintento
   correctivo y log correlacionado por `request_id`.
3. **Baseline RAGAS** sobre cinco consultas con respuesta de referencia: `faithfulness`,
   `answer_relevancy`, `context_precision` y `context_recall`.

Al terminar tendrás:

- `SourceReference` y los campos `grounded`/`citations` en el esquema de estimación, sin romper los
  contratos existentes.
- `verify_citations` en `app/generation/rag/grounding.py` y su integración en el orquestador, en el
  endpoint por etapas y en el consenso de horas.
- Un golden set de generación (`evals/golden_generation_s11.json`) con `ground_truth` por consulta.
- Un harness de recogida y scoring RAGAS (`scripts/eval_ragas_s11.py` +
  `scripts/score_ragas_s11.py`) y la tabla de 4 métricas × 5 consultas + promedio.
- `docs/sesion-11-citacion-verificable-ragas.md` con la evidencia real de la ejecución y una nota
  breve sobre lo que más chirría de los números.

> **Alcance estricto.** Esta sesión **no** cambia el retrieval, ni añade fuentes, ni reescribe el
> wizard, ni introduce un modelo de estimación nuevo. Envuelve el pipeline existente con citas
> verificables y lo mide. No se toca la base de datos: los ids de chunk ya son la clave primaria y
> el `budget_id` ya viaja en la metadata y en `RetrievedChunk`.

## Reconciliación con el repo

El punto de partida es `main` con la Sesión 10 cerrada: retrieval híbrido, multifuente, enrutado,
plan de consulta, vigencia, reranking, estructura sin horas y consenso de horas por tarea.

**Nada de eso se reescribe.** En particular:

- La arquitectura por capas manda. El código de generación vive en `app/generation/rag/`, las
  plantillas en `app/foundation/prompts/`, el transporte HTTP en `app/api/`, la configuración en
  `app/config.py` y el *composition root* en `app/dependencies.py`.
- El contrato de estimación se llama `GroundedEstimate`; la línea es `EstimationTask`; el bloque es
  `EstimationModule`. **No se renombran** a `Estimate`, `TaskItem` ni `WorkModule`. La tabla de
  equivalencias con el repo de referencia es:

  | Repo de referencia (`session_11`) | Este repo |
  |---|---|
  | `Estimate` | `GroundedEstimate` |
  | `TaskItem` | `EstimationTask` |
  | `WorkModule` | `EstimationModule` |
  | `SourceCitation` | `Citation` |
  | `context_assembler.py` | `context_builder.py` |
  | `validation.py` | `grounding.py` |
  | `prompt_builder.py` | plantillas Jinja versionadas en `app/foundation/prompts/` |

- El generador **no** usa `client.responses.parse` crudo. El equivalente en este repo es
  `LLMWrapper.complete_structured` (salida estructurada Pydantic, sin prosa ni fences), que ya es
  el camino de `generate_grounded_estimate` y `generate_structure`. La spec mantiene esa pieza; el
  schema estricto se consigue con Pydantic v2 y `response_model`, no con una API nueva.
- La generación estructura-primero de la Sesión 10 (`rag_structure/v1`) **no tiene retrieval**. Sus
  líneas no pueden citar fuentes: nacen con `grounded=False`. El consenso de horas
  (`task_hours.estimate_one`) es quien aporta la evidencia y las citas reales por tarea.
- `EstimationTask.sources: list[int]` y `TaskHoursEstimate.has_match` **se conservan** por
  compatibilidad con tests, cliente web y endpoints. Los campos nuevos son aditivos:
  `grounded`, `citations` (en `EstimationTask`), `grounded` y `sources` (en `TaskHoursEstimate`).
  Una línea nueva puede llevar ambas representaciones; `sources` se sincroniza desde `citations`
  para que los consumidores antiguos sigan funcionando.
- Los errores RAG viven en `app/generation/rag/errors.py`. No se importan errores de otro proyecto.
- El corpus canónico está en **inglés** (`data/budgets_sample.json`, `data/task_corpus.json`). Los
  nuevos prompts y todo el código van en inglés. La prosa de la nota y del golden set puede ir en
  español; el `ground_truth` de esta spec se escribe en inglés para que el juez compare en el mismo
  idioma que la respuesta generada.
- El golden set de retrieval de la Sesión 10 vive en `evals/golden_retrieval.json` y ya incluye
  Q1–Q8. Esta sesión **no lo modifica**: crea `evals/golden_generation_s11.json` a partir de las
  cinco consultas budget-only Q1–Q5 y les añade `transcript` y `ground_truth`.
- El harness de evaluación corre en dos pasos porque el stack de `langchain` del proyecto es más
  nuevo que el que fija RAGAS. La recogida de muestras usa el venv del proyecto; el scoring usa un
  venv aislado con RAGAS. Esto es deliberado y está documentado en el Paso 0.

### Piezas que se reutilizan

- `app/generation/rag/schemas.py` — `GroundedEstimate`, `EstimationTask`, `EstimationModule`,
  `RetrievedChunk`, `Citation`, `TaskHoursEstimate`.
- `app/generation/rag/grounding.py` — `find_fabricated_citations` e `is_coherent_estimate`. Aquí se
  añade `verify_citations`.
- `app/generation/rag/context_builder.py` — `render_source`, `assemble_context`,
  `fit_chunks_to_budget`.
- `app/generation/rag/grounded_estimator.py` — orquestador `transcript → brief → retrieval →
  contexto → generación → validación`.
- `app/generation/rag/task_hours.py` — consenso de horas y `TaskNeighbor`.
- `app/foundation/prompts/loader.py` y las plantillas `rag_estimation/{v1,v2}` y
  `rag_structure/v1`.
- `app/api/rag_estimation.py` — pipeline completo y etapas del wizard.
- `app/api/estimate_tasks.py` — `POST /api/v1/estimate/tasks/hours`.
- `app/generation/rag/stage_logging.py` — `current_request_id` y `log_pipeline_stage`.
- `app/foundation/llm/` — `LLMWrapper.complete_structured`.
- `app/dependencies.py` — `get_semantic_retriever`, `get_token_encoder`,
  `get_runtime_retrieval_config`.
- `scripts/rag_ops/common.py` — utilidades de ejecución y `require_embedder`.
- `tests/test_grounding.py`, `tests/test_rag_schemas.py`, `tests/test_task_hours.py`,
  `tests/test_rag_stages.py` — estilo de tests con dobles, sin Postgres ni red.

## Requisitos para el ejercicio

- Proyecto de la Sesión 10 en verde:

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

  El servicio `api` aplica `alembic upgrade head` al arrancar. Si `/health` no responde, no
  empieces.

- Corpus base ingerido de forma idempotente:

  ```bash
  docker compose run --rm api python scripts/query_examples.py
  ```

  Debes tener los **17 presupuestos / 60 chunks** de `data/budgets_sample.json`. Para el consenso de
  horas hace falta además el corpus task-level:

  ```bash
  docker compose run --rm api python scripts/build_task_corpus.py --ingest
  ```

  El golden set de esta sesión se evalúa solo sobre `chunk_type="budget_component"`, para que la
  tabla RAGAS sea reproducible aunque el corpus task-level esté ingerido.

- Clave de OpenAI en `.env` (`OPEN_AI_KEY`) o un endpoint OpenAI-compatible. RAGAS necesita,
  además, una clave para el juez y para los embeddings: `OPENAI_API_KEY` o `OPEN_AI_KEY`.

- Dependencias de scoring en un **venv aislado** (ver Paso 0.1). No añadas `ragas` a
  `pyproject.toml`: su rango de `langchain-core` choca con el `langchain-core` actual del proyecto.

- Docker y Docker Compose funcionando. El modelo de embeddings debe ser el mismo de la ingesta
  (`text-embedding-3-small`).

## Lo que entra y lo que no entra

**Entra:**

- `SourceReference`, `LineCitation` y `CitationReport` en `app/generation/rag/schemas.py`.
- Campos `grounded` y `citations` en `EstimationTask`; `grounded` y `sources` en
  `TaskHoursEstimate`.
- Validador de integridad: `grounded=True` exige al menos una fuente; `grounded=False` no puede
  llevar horas inventadas ni fuentes.
- Atributo `document_id` en los bloques `<source>` y prompts de atribución por línea.
- `verify_citations(estimate, retrieved_chunk_ids)` y su variante
  `verify_citations_for_chunks(estimate, chunks)`.
- Integración en el pipeline completo, en `POST /api/v1/estimate/stages/generate` y en el endpoint
  de horas.
- Demo offline con una citación colgante plantada a propósito.
- Golden set de generación con `ground_truth` por consulta.
- Harness RAGAS de recogida y scoring y tabla de métricas.
- Tests unitarios de esquema, verificación, consenso y endpoint.
- Documento de resultados en `docs/`.

**No entra:**

- Cambios en retrieval, reranking, multifuente, enrutado, plan de consulta o vigencia.
- Cambios en el corpus o re-ingesta masiva.
- Un modelo de estimación nuevo, un endpoint nuevo o una migración Alembic.
- Cambios en la UI, en `cliente-web/` o en el wizard.
- Fine-tuning del juez, comparación de modelos de RAGAS o métricas adicionales.
- Cambiar el contrato HTTP de forma incompatible: `sources` y `has_match` siguen existiendo.
- Copiar literalmente el repo de referencia. La spec adapta nombres, rutas y contratos a este repo.

## Datos para el ejercicio

### Golden set de generación

`evals/golden_generation_s11.json` parte de las cinco consultas budget-only de
`evals/golden_retrieval.json` y añade:

- `transcript`: notas de reunión realistas que se pasan al pipeline.
- `ground_truth`: estimación de referencia que un experto produciría con los presupuestos
  históricos relevantes, en **engineer-days** (horas / 8).
- `relevant_budget_ids`: los mismos presupuestos anotados en la Sesión 10.

```json
{
  "description": "Session 11 golden generation set. Extends the Session 10 retrieval golden set (evals/golden_retrieval.json, queries Q1-Q5, budget-only) with an expert reference estimate (ground_truth) per query, so RAGAS can score the generated estimate against a reference. question = the estimation request; transcript = the meeting notes fed to the pipeline; ground_truth = the reference estimate (engineer-days, ~8h/day) an expert would produce from the relevant historical budgets. Engineer-days are derived from the recorded estimated_hours of the relevant budgets in data/budgets_sample.json.",
  "source": "evals/golden_retrieval.json (Q1-Q5)",
  "corpus": "data/budgets_sample.json",
  "judge_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "queries": [
    {
      "id": "Q1",
      "query": "Mobile banking application with OAuth2 authentication, PSD2 open banking compliance, strong customer authentication and a transaction ledger.",
      "transcript": "Client kickoff call. They are a neobank and want a mobile banking application. Must-haves: OAuth 2.0 authentication backend, PSD2 open banking connectors to third-party banks, strong customer authentication (SCA) on logins and payments, and a transaction ledger service that records every movement. They also want push and email notifications for transactions. Stack is a Python backend with a React Native app. Compliance with PSD2 is non-negotiable; this is the regulated EU financial sector.",
      "relevant_budget_ids": ["BUD-2024-001", "BUD-2024-003"],
      "ground_truth": "Reference estimate ~60 engineer-days, grounded mainly in BUD-2024-001 (mobile banking API, 480h). Modules: Authentication & SCA — OAuth 2.0 backend plus strong customer authentication, ~15 engineer-days (AUTH-001, 120h). PSD2 & Open Banking — open banking connectors, ~20 engineer-days (PSD2-002, 160h). Transaction Ledger — ledger service, ~18 engineer-days (TXN-003, 140h). Notifications — push and email, ~8 engineer-days (NOTIF-004, 60h). Fraud/payments hardening can be informed by BUD-2024-003 as a supporting source. Confidence high: the project maps almost one-to-one to a historical banking budget."
    },
    {
      "id": "Q2",
      "query": "Headless e-commerce storefront with a product catalog, shopping cart checkout and personalized product recommendations.",
      "transcript": "Retail client wants a headless e-commerce storefront. Scope: a product catalog API, a cart and checkout service, and personalized product recommendations on the storefront. The front end is a PWA storefront. They are explicitly not building a marketplace and not handling multi-vendor payouts. Medium scale, B2C, EU. Stack is a Node backend with a React PWA.",
      "relevant_budget_ids": ["BUD-2024-005", "BUD-2024-006", "BUD-2024-007", "BUD-2024-017"],
      "ground_truth": "Reference estimate ~59 engineer-days, grounded mainly in BUD-2024-005 (headless e-commerce storefront, 460h). Modules: Catalog — product catalog API, ~19 engineer-days (CATALOG-001, 150h). Cart & Checkout — cart and checkout service, ~18 engineer-days (CART-002, 140h). Recommendations — personalized recommendations, ~14 engineer-days (RECO-003, 110h). Storefront — storefront PWA, ~8 engineer-days (STORE-004, 60h). BUD-2024-006/007/017 are supporting analogs for catalog/checkout patterns. Confidence high. Vendor payouts and dispute handling from BUD-2024-006 are out of scope and must NOT be added."
    },
    {
      "id": "Q3",
      "query": "Telemedicine platform for patient appointment scheduling and video consultations with HL7/FHIR integration and clinical records.",
      "transcript": "Healthcare provider wants a telemedicine platform. Requirements: patient appointment scheduling, video consultations between patients and clinicians, HL7/FHIR integration with their existing hospital systems, and clinical records with consent and audit logging. Regulated healthcare sector, EU, GDPR plus clinical data handling. Stack is a Python backend.",
      "relevant_budget_ids": ["BUD-2024-009", "BUD-2024-010"],
      "ground_truth": "Reference estimate ~70 engineer-days, grounded mainly in BUD-2024-009 (telemedicine platform with HL7/FHIR, 560h). Modules: FHIR Integration — integration layer, ~24 engineer-days (FHIR-001, 190h). Scheduling — appointment scheduling, ~19 engineer-days (APPT-002, 150h). Telemedicine Video — video consultation sessions, ~20 engineer-days (VIDEO-003, 160h). Consent & Audit — consent and audit module for clinical records, ~8 engineer-days (CONSENT-004, 60h). BUD-2024-010 (remote patient monitoring) is a supporting source. Confidence high."
    },
    {
      "id": "Q4",
      "query": "Industrial IoT platform ingesting sensor telemetry for predictive maintenance and real-time equipment monitoring in a factory.",
      "transcript": "Manufacturing client wants an industrial IoT platform for a factory. Scope: ingest sensor telemetry from equipment, predictive maintenance models that flag failures before they happen, and real-time equipment monitoring dashboards integrated with their SCADA system. Industrial sector, on-premise plus cloud. High data volume.",
      "relevant_budget_ids": ["BUD-2024-013", "BUD-2024-015"],
      "ground_truth": "Reference estimate ~62 engineer-days, grounded mainly in BUD-2024-013 (factory IoT telemetry and predictive maintenance, 500h). Modules: Telemetry Ingestion — sensor telemetry ingestion, ~21 engineer-days (TELE-001, 170h). Predictive Maintenance — predictive maintenance models, ~25 engineer-days (PRED-002, 200h). SCADA Dashboard — SCADA dashboard integration for real-time monitoring, ~16 engineer-days (SCADA-003, 130h). BUD-2024-015 (energy grid monitoring and demand-response) is a supporting analog for the data-pipeline and forecasting parts. Confidence high/medium."
    },
    {
      "id": "Q5",
      "query": "Real-time payments gateway with fraud detection, transaction reconciliation and a double-entry ledger.",
      "transcript": "Fintech client wants a real-time payments gateway. Requirements: a payment gateway core processing transactions in real time, a real-time fraud detection engine, settlement and transaction reconciliation, and a double-entry ledger for accounting integrity. They also want a merchant dashboard. Regulated EU financial sector. Stack is a Python backend.",
      "relevant_budget_ids": ["BUD-2024-003", "BUD-2024-001"],
      "ground_truth": "Reference estimate ~76 engineer-days, grounded mainly in BUD-2024-003 (real-time payments gateway with fraud detection and reconciliation, 600h). Modules: Gateway Core — payment gateway core, ~28 engineer-days (GATE-001, 220h). Fraud Detection — real-time fraud detection, ~25 engineer-days (FRAUD-002, 200h). Reconciliation & Ledger — settlement reconciliation and the double-entry ledger, ~15 engineer-days (RECON-003, 120h). Merchant Dashboard — ~8 engineer-days (DASH-004, 60h). BUD-2024-001 (mobile banking API with a transaction ledger) is a supporting source for the ledger. Confidence high."
    }
  ]
}
```

> **No copies el golden set sin mirarlo.** Las anotaciones salen del corpus de este repo
> (`data/budgets_sample.json`). Comprueba que los `relevant_budget_ids` y las horas citadas existen
> antes de usarlo. El `ground_truth` es una referencia de experto, no una verdad absoluta: RAGAS lo
> usa para medir `context_recall`, y una discrepancia razonable de módulos es aceptable.

### Unidades

El corpus guarda **horas**; la estimación trabaja en **engineer-days**. La convención de esta sesión
es `engineer_days = estimated_hours / 8`, redondeado. El `ground_truth` ya está en engineer-days.
Si el modelo copia horas como días, la tabla lo delatará: es uno de los artefactos que la nota final
debe comentar.

### Tabla de referencia (no sustituye tu ejecución)

La siguiente tabla es la salida observada en el repo `session_11` con el mismo juez y embeddings.
Sirve para detectar desviaciones groseras, no para copiarla. Tu entregable debe tener **tu** salida
real.

| query | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| Q1 | 0.418 | 0.000 | 1.000 | 0.000 |
| Q2 | 0.426 | 0.061 | 1.000 | 0.000 |
| Q3 | 0.564 | 0.000 | 1.000 | 0.571 |
| Q4 | 0.877 | 0.000 | 1.000 | 0.000 |
| Q5 | 0.476 | 0.104 | 1.000 | 0.000 |
| **average** | 0.552 | 0.033 | 1.000 | 0.114 |

## ✍ Ejercicio

### Paso 0 — Gate de verificación y entorno de RAGAS

#### 0.1 — Venv aislado para RAGAS

El proyecto usa `langchain-core>=1.7` y `langchain-community>=0.4`. RAGAS fija versiones más
antiguas y, al importarse, toca módulos de Vertex AI que el `langchain-community` actual ya no
expone. Por eso el scoring va en un venv separado:

```bash
uv venv .venv-ragas --python 3.12
uv pip install --python .venv-ragas/bin/python \
  "ragas>=0.4,<0.5" \
  datasets \
  langchain-openai
```

Comprueba el gate:

```bash
.venv-ragas/bin/python -c "import ragas, datasets, langchain_openai; print('ragas ok')"
```

Si el import de RAGAS falla por Vertex, el script del Paso 5.3 instala los stubs necesarios. No
metas RAGAS en `pyproject.toml`: rompería la resolución del proyecto.

El venv no se commitea. Añade `.venv-ragas/` a `.gitignore` si no está ya cubierto por `.venv*`.

#### 0.2 — Baseline del proyecto

Antes de tocar código, deja el repo en verde:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app frontend
```

Si algo falla, arréglalo primero. Esta sesión añade campos a modelos Pydantic que usan tests,
endpoints y el cliente web; un baseline rojo hace imposible distinguir una regresión nueva.

### Paso 1 — Schema extendido

En `app/generation/rag/schemas.py`, añade `model_validator` a la importación de Pydantic:

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator
```

#### 1.1 — `SourceReference`

Añádelo junto a `Citation` y `Assumption`:

```python
class SourceReference(BaseModel):
    """A verifiable, line-level citation back to a single retrieved chunk.

    Session 11 lifts citation from the coarse, estimate-global ``Citation``
    down to the individual estimate line: every grounded line carries the
    concrete chunk it derived from plus the verbatim evidence that backs it,
    so a reader (and ``grounding.verify_citations``) can check the claim
    instead of trusting it.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        description="Id of the retrieved chunk supporting this line — the `id` "
        "attribute of the <source> element it was taken from."
    )
    document_id: str = Field(
        description="Traceable parent document (budget_id for budget/task chunks, "
        "source_id for transcripts and technical docs) — the `document_id` "
        "attribute of the same <source> element."
    )
    evidence: str = Field(
        min_length=1,
        description="Verbatim span or figure copied from the source that backs "
        "this line. Quote the source; do not paraphrase.",
    )
```

`document_id` es el **presupuesto histórico concreto**, no la clave primaria de la tabla
`documents`. En los chunks de presupuesto y de tarea vale `BUD-2024-XXX` o `TASK-2022-XXXX`; en
transcripciones y documentación técnica vale su `source_id`. El helper
`context_builder.traceable_document_id` centraliza esa regla.

#### 1.2 — `EstimationTask`

Sustituye la clase actual por esta versión. Los campos nuevos son **aditivos**: `sources` sigue
existiendo para los consumidores de la Sesión 09.

```python
class EstimationTask(BaseModel):
    """Una tarea concreta dentro de un módulo funcional, en engineer-days.

    Each task is an estimate *line*. When ``grounded`` is True the line derives
    from historical evidence and carries at least one ``SourceReference``; when
    False the line has no sufficient source data — it must not invent hours and
    should surface its scope as an ``Assumption`` instead.

    ``sources`` is the legacy int-id citation from Session 9. It is kept for
    backward compatibility and synchronised from ``citations`` after validation;
    new generators populate ``citations`` only.
    """

    name: str
    description: str | None = Field(default=None, description="Alcance en una línea.")
    engineer_days: int | None = Field(
        default=None,
        ge=0,
        description="Effort in engineer-days. None in the structure-only generation "
        "mode (Session 10): the LLM proposes the module→task structure and the hours "
        "are derived afterwards by per-task vector search, not inferred here. Must be "
        "None when ``grounded`` is False (no source data => no invented hours).",
    )
    grounded: bool = Field(
        default=False,
        description="True iff this line is backed by retrieved source data. False "
        "means 'no sufficient source data': the line carries no sources and no "
        "invented hours.",
    )
    citations: list[SourceReference] = Field(
        default_factory=list,
        description="Per-line citations. Non-empty iff ``grounded`` is True.",
    )
    sources: list[int] = Field(
        default_factory=list,
        description="Legacy chunk ids from Session 9. Do not populate in new prompts; "
        "it is synchronised from ``citations`` after validation.",
    )

    @model_validator(mode="after")
    def _grounding_integrity(self) -> "EstimationTask":
        """Enforce the Session 11 line-level grounding contract.

        A grounded line must cite at least one source; an ungrounded line must
        neither cite sources nor invent hours. Legacy objects built by Session 9
        tests with only ``sources`` and ``engineer_days`` are tolerated: the
        validator only activates when the new fields are explicitly provided.
        """
        if "grounded" not in self.model_fields_set and not self.citations:
            return self
        if self.grounded and not self.citations:
            raise ValueError("a grounded line must cite at least one source")
        if not self.grounded and self.citations:
            raise ValueError("an ungrounded line (grounded=False) must not carry citations")
        if not self.grounded and self.sources:
            raise ValueError("an ungrounded line (grounded=False) must not carry legacy sources")
        if not self.grounded and self.engineer_days is not None:
            raise ValueError(
                "an ungrounded line (grounded=False) must leave engineer_days null; "
                "do not invent hours without source data"
            )
        if self.citations:
            self.sources = [
                int(citation.chunk_id)
                for citation in self.citations
                if citation.chunk_id.lstrip("-").isdigit()
            ]
        return self
```

#### 1.3 — `TaskHoursEstimate`

El consenso de horas ya devuelve `neighbors`; ahora devuelve también las fuentes verificables que
alimentaron la cifra. `has_match` se conserva y `grounded` lo refleja.

```python
class TaskHoursEstimate(BaseModel):
    """Horas derivadas para una tarea desde el corpus histórico.

    ``has_match=False`` (ningún vecino bajo el umbral) deja ``estimated_hours`` y
    ``reliability`` a null: la UI marca la fila en rojo y la persona aporta el
    número en el paso de validación.

    Session 11 adds ``grounded`` and ``sources``: the exact historical chunks
    whose recorded hours produced the consensus. ``grounded`` is the new name
    for "has a source-backed value"; ``has_match`` stays as the legacy flag.
    """

    module: str = Field(description="Módulo al que pertenece la tarea (eco).")
    task: str = Field(description="Nombre de la tarea (eco).")
    estimated_hours: int | None = Field(
        default=None, ge=0, description="Horas del consenso ponderado, o null si no hay match."
    )
    reliability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0..1 confianza en las horas (cercanía x acuerdo entre vecinos).",
    )
    has_match: bool = Field(description="Si algún vecino cruzó el umbral de distancia.")
    grounded: bool = Field(
        default=False,
        description="True iff the hours came from at least one retrieved historical "
        "chunk. Mirrors ``has_match``; the new pipeline sets both explicitly.",
    )
    dispersion: float | None = Field(
        default=None,
        ge=0.0,
        description="Dispersión de las horas de los vecinos (coeficiente de variación); "
        "más alto = menos acuerdo.",
    )
    neighbors: list[TaskNeighbor] = Field(
        default_factory=list, description="Tareas históricas de las que salieron las horas."
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Verbatim line-level citations built from the matching historical "
        "chunks. Non-empty iff ``grounded`` is True.",
    )

    @model_validator(mode="after")
    def _grounding_integrity(self) -> "TaskHoursEstimate":
        """Keep ``grounded``, ``has_match`` and ``sources`` consistent.

        Legacy fixtures that only set ``has_match`` are tolerated: ``grounded``
        is derived from it. The pipeline sets both fields explicitly and must
        provide sources for a grounded line.
        """
        if "grounded" not in self.model_fields_set:
            self.grounded = self.has_match
        if self.grounded != self.has_match:
            raise ValueError("grounded and has_match must agree")
        if self.grounded and not self.sources and "sources" in self.model_fields_set:
            raise ValueError("a grounded line must cite at least one source")
        if not self.grounded and self.sources:
            raise ValueError("an ungrounded line (grounded=False) must not carry sources")
        if not self.grounded and self.estimated_hours is not None:
            raise ValueError(
                "an ungrounded line (grounded=False) must leave estimated_hours null"
            )
        return self
```

#### 1.4 — Informe de verificación

Añade estos modelos al final del bloque de estimación de `schemas.py`, **antes** de los modelos HTTP
(`RetrievalQuery`, `TranscriptEstimateRequest`, `GroundedGenerationOutcome`). `GroundedGenerationOutcome`
usará `CitationReport` en el Paso 3.3; si lo declaras después, Python falla al importar el módulo.

```python
LineCitationStatus = Literal["grounded", "dangling", "insufficient"]


class LineCitation(BaseModel):
    """The citation status of a single estimate line."""

    module: str = Field(description="Module the line belongs to.")
    component: str = Field(description="The line (task) name.")
    status: LineCitationStatus = Field(
        description="grounded = all cited chunks were retrieved; dangling = at "
        "least one cited chunk_id was never in the context or points to the wrong "
        "document; insufficient = the line is flagged as having no sufficient "
        "source data."
    )
    cited_chunk_ids: list[str] = Field(default_factory=list)
    dangling_chunk_ids: list[str] = Field(
        default_factory=list, description="Cited ids that were never retrieved."
    )
    document_mismatches: list[str] = Field(
        default_factory=list,
        description="Citations whose chunk exists but whose document_id does not "
        "match the retrieved parent budget/document.",
    )
    reason: str | None = Field(
        default=None, description="Human-readable reason when status is not grounded."
    )


class CitationReport(BaseModel):
    """Aggregate result of verifying an estimate's citations against the context."""

    total_lines: int = Field(ge=0)
    grounded_lines: int = Field(ge=0, description="Lines whose every citation was retrieved.")
    dangling_lines: int = Field(ge=0, description="Lines with at least one fabricated citation.")
    insufficient_lines: int = Field(ge=0, description="Lines flagged as no sufficient data.")
    verified_citations: int = Field(
        ge=0, description="Count of individual citations that resolved to a retrieved chunk."
    )
    dangling_citations: list[str] = Field(
        default_factory=list,
        description="Sorted, de-duplicated cited ids that were never retrieved "
        "(line-level and estimate-global). Empty means every citation is real.",
    )
    dangling_global_citations: list[str] = Field(
        default_factory=list,
        description="Estimate-global Citation.source_id values that were never retrieved.",
    )
    document_mismatches: list[str] = Field(
        default_factory=list,
        description="Sorted, de-duplicated document mismatches across all lines.",
    )
    lines: list[LineCitation] = Field(default_factory=list)

    @property
    def has_dangling(self) -> bool:
        """True when any citation points outside the retrieved context.

        A document mismatch counts even when the chunk id exists, and a
        grounded line with no citations counts as dangling too.
        """
        return bool(self.dangling_citations or self.document_mismatches or self.dangling_lines)
```

> **Por qué `document_id` se verifica además del `chunk_id`.** Un id de chunk puede existir en el
> contexto y, aun así, la línea atribuirse al presupuesto equivocado. El objetivo de la sesión no es
> solo que la cita exista, sino que apunte al **presupuesto histórico concreto** del que se derivó.
> La verificación de documento convierte esa promesa en algo comprobable.

### Paso 2 — Contexto y prompts de atribución por línea

#### 2.1 — `<source>` con `document_id`

En `app/generation/rag/context_builder.py`, añade el helper y actualiza `render_source`:

```python
def traceable_document_id(chunk: RetrievedChunk) -> str:
    """Return the traceable parent document id for a retrieved chunk.

    Budget and historical-task chunks carry ``budget_id``; transcripts and
    technical docs carry ``source_id``. Falling back to the DB ``document_id``
    keeps the value non-empty for any future collection.
    """
    return chunk.budget_id or chunk.source_id or str(chunk.document_id)


def render_source(chunk: RetrievedChunk) -> str:
    """Renderiza un chunk como elemento ``<source>`` autodescriptivo."""
    document_id = traceable_document_id(chunk)
    return (
        f'<source id="{chunk.id}" document_id="{document_id}" '
        f'sector="{chunk.sector}" project_year="{chunk.project_year}" '
        f'chunk_type="{chunk.chunk_type}" distance="{chunk.distance:.4f}">\n'
        f"{chunk.content}\n"
        f"</source>"
    )
```

El modelo ya puede copiar el `id` y el `document_id` exactos. Sin ese atributo, el prompt de
atribución no tiene de dónde sacar el presupuesto padre.

#### 2.2 — `rag_estimation/v3` (con horas)

Crea `app/foundation/prompts/rag_estimation/v3/system.j2`. Es la versión de la Sesión 11 del prompt
con horas: mantiene la estructura de v1 y añade la citación verificable por línea.

```jinja
You are a senior software-delivery estimator. Produce a cost estimate in engineer-days for the
project described in the user message, grounded ONLY in the historical budgets supplied as
<source> blocks.

## Structure

Organise the estimate as functional MODULES, each decomposed into the concrete engineering TASKS
needed to deliver it:

- Use the functional blocks that fit THIS project (for example Authentication & Access, Payments &
  Billing, Core Domain, Data & Integrations, Frontend/UX, Infrastructure & DevOps, Security &
  Compliance, QA & Testing, Project Management). Omit blocks that do not apply.
- Inside each module, break the work into granular tasks. Give every task a short `description` and
  its own `engineer_days`. Aim for a thorough breakdown (typically 4-8 modules with several tasks
  each), not a handful of coarse line items.
- A historical <source> component often maps to a module or a small group of tasks: decompose it
  into the finer tasks a delivery team would actually plan and distribute its engineer-days across
  them.

## Grounding rules

1. Base every number ONLY on the <source> blocks provided. Do not use outside knowledge for
   quantities.
2. Every <source> block carries an `id` and a `document_id` attribute. Per-LINE citation is
   mandatory and must be verifiable. For each task derived from historical evidence, set
   `grounded=true` and fill its `citations` list with one entry per source used:
   - `chunk_id`: the exact `id` attribute of the <source> element.
   - `document_id`: the exact `document_id` attribute of that same <source>.
   - `evidence`: a VERBATIM span or figure copied from that source (for example the component name
     and its estimated hours). Do not paraphrase.
3. Only cite ids that literally appear as an `id` attribute in the <sources> block. Never invent or
   guess a chunk_id or document_id.
4. If a task has no sufficient support in the sources, set `grounded=false`, leave `citations`
   empty and `engineer_days` null. Do NOT estimate its hours by eye. Capture that scope in
   `assumptions` instead.
5. Leave the legacy `sources` field empty. `citations` is the authoritative field for this session.
6. `total_engineer_days` must equal the sum of the `engineer_days` of all grounded tasks across all
   modules.
7. If the context is insufficient to estimate responsibly, set `confidence="insufficient"`, leave
   `total_engineer_days` and `duration_weeks` null, leave `modules` empty, and explain what is
   missing in `insufficient_context_explanation`.
8. Otherwise set `confidence` to high/medium/low according to how well the sources match the
   project, and explain the derivation in `reasoning`.

Return a single valid JSON object that matches the required schema. Include every field; use null or
[] where appropriate. Do not wrap the JSON in prose or Markdown fences.
```

`app/foundation/prompts/rag_estimation/v3/user.j2` reutiliza el de v1:

```jinja
<project_brief>
Functionality: {{ brief.functionality }}
Technologies: {{ brief.technologies | join(', ') if brief.technologies else 'n/a' }}
Sector: {{ brief.sector or 'n/a' }}
Scale: {{ brief.scale }}
Country: {{ brief.country or 'n/a' }}
Regulations: {{ brief.regulations | join(', ') if brief.regulations else 'n/a' }}
Constraints: {{ brief.constraints | join(', ') if brief.constraints else 'n/a' }}
</project_brief>

<sources>
{{ context_block }}
</sources>

Produce the grounded estimate now. Every grounded task must cite at least one real source with
verbatim evidence.
```

#### 2.3 — `rag_estimation/v4` (estructura con contexto, sin horas)

Crea `app/foundation/prompts/rag_estimation/v4/system.j2` a partir de v2. Reglas clave:

```jinja
You are a senior software-delivery estimator. Decompose the project described by the user into the
functional MODULES and the concrete engineering TASKS needed to deliver it, grounded in the
historical budgets supplied as <source> blocks. DO NOT estimate hours or effort: that is done in a
later step.

## Structure

Organise the work into functional blocks that fit THIS project. Inside each module, break the work
into granular tasks with a short `description`. Aim for a thorough breakdown (typically 4-8 modules
with several tasks each).

## Grounding rules

1. Leave `engineer_days` null for every task and `total_engineer_days` null: you are NOT estimating
   effort here.
2. Every <source> block carries an `id` and a `document_id`. For a task that maps to a historical
   component, set `grounded=true` and fill `citations` with the exact `chunk_id`, the exact
   `document_id`, and a VERBATIM `evidence` span copied from that source.
3. Only cite ids that literally appear in the <sources> block. Never invent ids.
4. Genuinely novel scope with no historical analog must be left `grounded=false` (empty
   `citations`) and expressed as an Assumption.
5. Leave the legacy `sources` field empty; `citations` is authoritative.
6. If the provided context is insufficient to scope the project responsibly, set
   `confidence="insufficient"`, leave `modules` empty and explain what is missing in
   `insufficient_context_explanation`.
7. Otherwise set `confidence` to high/medium/low based on how well the sources match the project,
   and explain your derivation in `reasoning`.

Return a single valid JSON object that matches the required schema. Include every field; use null or
[] where appropriate. Do not wrap the JSON in prose or Markdown fences.
```

`app/foundation/prompts/rag_estimation/v4/user.j2` es igual al de v2.

#### 2.4 — `rag_structure/v2` (estructura sin retrieval)

`generate_structure` no recibe contexto, así que todas sus líneas nacen `grounded=false`. Cambiar el
prompt de estructura para que no intente citar es importante: si el esquema exige `citations` para
`grounded=true`, el modelo no debe inventarlas.

Crea `app/foundation/prompts/rag_structure/v2/system.j2` en inglés:

```jinja
You are a senior software-delivery architect. Decompose the project described by the user into the
functional MODULES and the concrete engineering TASKS needed to deliver it. This is a STRUCTURE-ONLY
step: you do NOT estimate hours and you do NOT have historical sources. Rely on your own engineering
judgement about what the project entails.

## Structure

- Organise the work into functional blocks that fit THIS project. Add sector-specific ones when the
  brief calls for them; omit the ones that do not apply.
- Inside each module, break the work into granular tasks with a short `description`. Be thorough —
  typically 5-9 modules with several tasks each — so a delivery team could plan from it.
- Write module and task names in technical English. The historical corpus is in English and the
  later hours step compares both texts; mixing languages triggers false "no data" matches.

## Rules

1. Leave `engineer_days` null for every task and `total_engineer_days` null — hours are derived in a
   later step.
2. Leave `citations` empty and `grounded=false` for every task: there is no historical context here.
   Do not invent citations.
3. Leave the legacy `sources` field empty.
4. Use `assumptions` for scope you are inferring beyond the brief.
5. If the brief is too vague to scope responsibly, set `confidence="insufficient"`, leave `modules`
   empty and explain what is missing in `insufficient_context_explanation`; otherwise set
   `confidence` to high/medium/low based on how well-specified the brief is and explain your
   reasoning in `reasoning`.
6. Maximum 8 modules and between 3 and 8 tasks per module. Prioritise the tasks that change the
   effort; do not inflate the breakdown.

Return a single valid JSON object that matches the required schema. Include every field; use null or
[] where appropriate. Do not wrap the JSON in prose or Markdown fences.
```

`app/foundation/prompts/rag_structure/v2/user.j2` reutiliza el brief de v1.

#### 2.5 — Loader

En `app/foundation/prompts/loader.py`:

```python
def render_rag_estimation_prompt(
    *,
    brief: object,
    context_block: str,
    include_hours: bool = True,
    version: str | None = None,
) -> tuple[str, str]:
    """Render the (system, user) pair for grounded generation.

    v3 asks for hours and verifiable line-level citations. v4 is the
    structure-with-context variant without hours. v1/v2 stay available for
    rollback; they do not know the ``grounded``/``citations`` contract.
    """
    context = {
        "brief": brief,
        "context_block": context_block,
    }
    resolved_version = version or ("v3" if include_hours else "v4")
    ...
```

```python
def render_rag_structure_prompt(brief: object, version: str = "v2") -> tuple[str, str]:
    """Prompt de la descomposición estructura-primero (sin horas, sin fuentes)."""
    return _render_rag_pair(
        RAG_STRUCTURE_USE_CASE,
        context={"brief": brief},
        version=version,
    )
```

> **No borres v1/v2.** El versionado de prompts existe para poder volver atrás sin tocar código.
> La sesión añade versiones; no reescribe las anteriores. `rag_structure/v1` se queda como
> referencia histórica.

### Paso 3 — Verificación de citaciones

#### 3.1 — `verify_citations` en `grounding.py`

Sustituye el contenido de `app/generation/rag/grounding.py` por esta versión. Mantiene
`find_fabricated_citations` e `is_coherent_estimate` para no romper los tests existentes.

```python
"""Post-generation checks for grounded estimates (Sessions 9 & 11).

Two independent guards run after the LLM returns a :class:`GroundedEstimate`:

* :func:`verify_citations` (Session 11) — line-level, programmatic citation
  verification. Every cited ``chunk_id`` (per-line :class:`SourceReference` and
  the estimate-global :class:`Citation`) must resolve to a chunk that was
  actually retrieved. Ids that were never in the context are *dangling*: a
  hallucination wearing the costume of rigor. The returned
  :class:`CitationReport` separates correctly grounded lines, dangling
  citations and lines explicitly marked as having no sufficient source data.
* :func:`is_coherent_estimate` — the ``insufficient`` confidence level has a
  strict shape (no numbers, an explanation present); a violation is a malformed
  response, not a valid estimate.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.generation.rag.context_builder import traceable_document_id
from app.generation.rag.schemas import (
    CitationReport,
    GroundedEstimate,
    LineCitation,
    RetrievedChunk,
)


def verify_citations(
    estimate: GroundedEstimate,
    retrieved_chunk_ids: set[str],
    *,
    retrieved_documents: Mapping[str, str] | None = None,
) -> CitationReport:
    """Verify every citation against the chunks actually handed to the LLM.

    Walks each estimate line (``modules[].tasks[]``) and checks that every
    ``chunk_id`` it cites is present in ``retrieved_chunk_ids``. When
    ``retrieved_documents`` is provided, the cited ``document_id`` must also
    match the traceable parent of the retrieved chunk; a chunk that exists but
    points to the wrong budget is just as misleading as an invented id.

    The estimate-global ``sources`` (Session 9 :class:`Citation`) are checked
    too, so a fabricated id cannot hide at either level.

    Parameters
    ----------
    estimate:
        The generated estimate to audit.
    retrieved_chunk_ids:
        The ids of the chunks that were actually placed in the context block
        (typically ``{str(chunk.id) for chunk in kept}``).
    retrieved_documents:
        Optional mapping ``chunk_id -> traceable document id`` used to verify
        that each citation names the right historical budget.

    Returns
    -------
    CitationReport
        Per-line statuses plus aggregate counts. ``dangling_citations`` is the
        sorted, de-duplicated set of cited ids that were never retrieved; an
        empty list means every citation is real.
    """
    lines: list[LineCitation] = []
    verified = 0
    dangling: set[str] = set()
    mismatches: set[str] = set()
    grounded_lines = dangling_lines = insufficient_lines = 0

    for module in estimate.modules:
        for task in module.tasks:
            explicit_grounding = "grounded" in task.model_fields_set
            cited = [ref.chunk_id for ref in task.citations]
            if not cited and task.sources:
                cited = [str(source_id) for source_id in task.sources]
            grounded = task.grounded if explicit_grounding else bool(cited)

            line_dangling = [cid for cid in cited if cid not in retrieved_chunk_ids]
            line_mismatches: list[str] = []
            if retrieved_documents is not None:
                for reference in task.citations:
                    expected = retrieved_documents.get(reference.chunk_id)
                    if expected is not None and reference.document_id != expected:
                        line_mismatches.append(
                            f"{reference.chunk_id}: cited document_id="
                            f"{reference.document_id!r} expected={expected!r}"
                        )

            verified += len(cited) - len(line_dangling)
            dangling.update(line_dangling)
            mismatches.update(line_mismatches)

            if not grounded:
                status = "insufficient"
                insufficient_lines += 1
                reason = "no sufficient source data"
            elif not cited:
                status = "dangling"
                dangling_lines += 1
                reason = "grounded line has no citations"
            elif line_dangling or line_mismatches:
                status = "dangling"
                dangling_lines += 1
                reason = "cited chunk not in context or wrong document"
            else:
                status = "grounded"
                grounded_lines += 1
                reason = None

            lines.append(
                LineCitation(
                    module=module.name,
                    component=task.name,
                    status=status,
                    cited_chunk_ids=cited,
                    dangling_chunk_ids=line_dangling,
                    document_mismatches=line_mismatches,
                    reason=reason,
                )
            )

    # Estimate-global citations (the coarse Session 9 layer) are verified too:
    # a fabricated id there is just as much a grounding failure.
    global_dangling: list[str] = []
    for citation in estimate.sources:
        chunk_id = str(citation.source_id)
        if chunk_id in retrieved_chunk_ids:
            verified += 1
        else:
            global_dangling.append(chunk_id)
    dangling.update(global_dangling)

    return CitationReport(
        total_lines=len(lines),
        grounded_lines=grounded_lines,
        dangling_lines=dangling_lines,
        insufficient_lines=insufficient_lines,
        verified_citations=verified,
        dangling_citations=sorted(dangling),
        dangling_global_citations=sorted(global_dangling),
        document_mismatches=sorted(mismatches),
        lines=lines,
    )


def verify_citations_for_chunks(
    estimate: GroundedEstimate,
    retrieved_chunks: list[RetrievedChunk],
) -> CitationReport:
    """Convenience wrapper: derive ids and traceable documents from the chunks."""
    return verify_citations(
        estimate,
        {str(chunk.id) for chunk in retrieved_chunks},
        retrieved_documents={
            str(chunk.id): traceable_document_id(chunk) for chunk in retrieved_chunks
        },
    )


def find_fabricated_citations(
    estimate: GroundedEstimate,
    retrieved_chunks: list[RetrievedChunk],
) -> list[int]:
    """Legacy helper: return cited int ids that were never retrieved.

    Kept for the Session 9 tests and any external consumer. New code should use
    :func:`verify_citations_for_chunks` and read the full report.
    """
    report = verify_citations_for_chunks(estimate, retrieved_chunks)
    return sorted(
        int(chunk_id) for chunk_id in report.dangling_citations if chunk_id.lstrip("-").isdigit()
    )


def is_coherent_estimate(estimate: GroundedEstimate) -> bool:
    """Comprueba que el nivel de confianza encaja con el contenido.

    Cuando ``confidence == "insufficient"``: ambos totales numéricos deben ser
    ``None``, ``modules`` debe estar vacío y la explicación de insuficiencia
    debe ser no vacía. Cualquier otro nivel se considera coherente aquí; las
    comprobaciones numéricas pertenecen al esquema y a las reglas de negocio.
    """
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )
```

> **Cuidado con la importación circular.** `grounding.py` importa
> `context_builder.traceable_document_id`; `context_builder.py` solo importa `schemas`. No hay ciclo.
> Si en el futuro `context_builder` importa `grounding`, mueve `traceable_document_id` a un módulo
> de utilidades compartido.

#### 3.2 — Integración en el pipeline completo

En `app/generation/rag/grounded_estimator.py`, sustituye la validación de citas del paso 5:

```python
from app.generation.rag.grounding import (
    is_coherent_estimate,
    verify_citations_for_chunks,
)

# ...

# 5. Line-level citation verification; one corrective retry on dangling ids.
report = verify_citations_for_chunks(estimate, kept)
log.info(
    "citations.verified",
    request_id=request_id,
    attempt=1,
    total_lines=report.total_lines,
    grounded=report.grounded_lines,
    dangling=report.dangling_lines,
    insufficient=report.insufficient_lines,
    dangling_ids=report.dangling_citations,
)
if report.has_dangling:
    feedback = (
        "Your previous answer cited sources that are not in the <sources> block or "
        f"attributed the wrong document_id: {report.dangling_citations + report.document_mismatches}. "
        "Copy only the id and document_id attributes that literally appear in <sources>, "
        "and use verbatim evidence from the same chunk."
    )
    with log_pipeline_stage("citation_retry", request_id, dangling=report.dangling_citations):
        estimate = await _generate(context_block, brief, feedback=feedback)
    report = verify_citations_for_chunks(estimate, kept)
    log.info(
        "citations.verified",
        request_id=request_id,
        attempt=2,
        total_lines=report.total_lines,
        grounded=report.grounded_lines,
        dangling=report.dangling_lines,
        insufficient=report.insufficient_lines,
        dangling_ids=report.dangling_citations,
    )
    if report.has_dangling:
        log.warning(
            "citations.unrepaired",
            request_id=request_id,
            dangling_ids=report.dangling_citations,
            document_mismatches=report.document_mismatches,
        )
        estimate = estimate.model_copy(update={"confidence": "low"})
```

Una citación colgante no se corrige silenciosamente: se registra con `citations.unrepaired` y baja
la confianza. El log va correlacionado con el mismo `request_id` que el resto de etapas.

#### 3.3 — Integración en el endpoint por etapas

En `app/api/rag_estimation.py`, importa la verificación:

```python
from app.generation.rag.grounding import (
    is_coherent_estimate,
    verify_citations_for_chunks,
)
```

En `structure`, añade el informe (todas las líneas serán `insufficient` porque no hay fuentes):

```python
citation_report = verify_citations_for_chunks(estimate, [])
return GroundedGenerationOutcome(
    estimate=estimate,
    fabricated_source_ids=[],
    citation_report=citation_report,
    coherent=True,
)
```

En `generate`, calcula el informe y mantén `fabricated_source_ids` por compatibilidad:

```python
citation_report = verify_citations_for_chunks(estimate, payload.kept_chunks)
return GroundedGenerationOutcome(
    estimate=estimate,
    fabricated_source_ids=[
        int(chunk_id)
        for chunk_id in citation_report.dangling_citations
        if chunk_id.lstrip("-").isdigit()
    ],
    citation_report=citation_report,
    coherent=is_coherent_estimate(estimate),
)
```

En `app/generation/rag/schemas.py`, amplía `GroundedGenerationOutcome`:

```python
class GroundedGenerationOutcome(BaseModel):
    """Salida de la generación: estimación + señales de grounding para el wizard."""

    estimate: GroundedEstimate
    fabricated_source_ids: list[int] = Field(
        default_factory=list,
        description="Source ids citados que no están en kept_chunks (vacío = limpio). "
        "Legacy mirror of citation_report.dangling_citations.",
    )
    citation_report: CitationReport | None = Field(
        default=None,
        description="Session 11 per-line citation verification report (None for the "
        "full from-transcript pipeline, which logs the report instead).",
    )
    coherent: bool = Field(
        description="False cuando una estimación insuficiente aún lleva números."
    )
```

#### 3.4 — Integración en el consenso de horas

En `app/generation/rag/task_hours.py`, añade un helper que convierta los chunks usados en
referencias verificables y rellena `grounded`/`sources`:

```python
from app.generation.rag.schemas import (
    RetrievedChunk,
    SourceReference,
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
    TaskNeighbor,
)


def _verbatim_evidence(chunk: RetrievedChunk) -> str:
    """Extract a short verbatim span from a historical chunk.

    The chunk text is already the verbatim source. We keep the component and
    hours lines when present, because they are the concrete evidence behind the
    consensus, and fall back to the whole chunk otherwise.
    """
    lines = [
        line
        for line in chunk.content.splitlines()
        if line.startswith(("Component:", "Estimated hours:", "Module:"))
    ]
    return "\n".join(lines).strip() or chunk.content.strip()[:280]


def _source_references(chunks: list[RetrievedChunk]) -> list[SourceReference]:
    """Build one verifiable citation per chunk that fed the hours consensus."""
    return [
        SourceReference(
            chunk_id=str(chunk.id),
            document_id=chunk.budget_id or chunk.source_id or str(chunk.document_id),
            evidence=_verbatim_evidence(chunk),
        )
        for chunk in chunks
    ]
```

Y actualiza las dos construcciones de `TaskHoursEstimate`:

```python
    if not usable:
        return TaskHoursEstimate(
            module=module,
            task=name,
            has_match=False,
            grounded=False,
            sources=[],
        )

    # ...

    return TaskHoursEstimate(
        module=module,
        task=name,
        estimated_hours=hours,
        reliability=reliability,
        has_match=True,
        grounded=True,
        dispersion=dispersion,
        neighbors=neighbors,
        sources=_source_references(usable),
    )
```

El endpoint `POST /api/v1/estimate/tasks/hours` devuelve ahora `sources` por tarea sin cambiar su
ruta ni su contrato de entrada. La UI puede ignorarlas o mostrarlas; el wizard existente sigue
funcionando.

#### 3.5 — Logging y visibilidad

La regla es simple: **toda citación colgante se loguea**. En el pipeline completo se emite
`citations.verified` en cada intento y `citations.unrepaired` si la reparación falla. En el endpoint
por etapas el informe viaja en la respuesta (`citation_report`) y el wizard puede pintarlo. No se
añade un endpoint nuevo ni se cambia el código HTTP de los errores.

### Paso 4 — Demo de citación colgante

Crea `scripts/demo_verify_citations_s11.py`. Es una prueba offline, sin red ni base de datos, que
construye una estimación realista, planta una cita colgante (`999`) y una línea sin datos, y
demuestra los tres criterios de aceptación de la Parte 1.

```python
#!/usr/bin/env python3
"""Offline demo of Session 11 line-level citation verification.

Builds a realistic grounded :class:`GroundedEstimate` from a small set of
"retrieved" chunks, deliberately plants ONE dangling citation (an id the LLM
never saw) and one line with no sufficient source data, then runs
:func:`verify_citations` and prints the :class:`CitationReport`. No network, no
database — it exercises the verification logic alone, so it doubles as the
acceptance-criteria proof:

* every grounded line cites at least one real chunk,
* the verifier flags the planted dangling citation,
* the unsupported line is reported as "insufficient", not back-filled.

Usage::

    uv run python scripts/demo_verify_citations_s11.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.generation.rag.grounding import verify_citations_for_chunks  # noqa: E402
from app.generation.rag.schemas import (  # noqa: E402
    Assumption,
    CitationReport,
    EstimationModule,
    EstimationTask,
    GroundedEstimate,
    RetrievedChunk,
    SourceReference,
)


def _retrieved() -> list[RetrievedChunk]:
    """The chunks the generator was actually given (ids 101, 102, 103)."""
    return [
        RetrievedChunk(
            id=101,
            document_id=1,
            content="BUD-2024-001 :: AUTH-001 OAuth 2.0 authentication backend — 120 h",
            sector="finance",
            project_year=2024,
            chunk_type="budget_component",
            distance=0.21,
            source_id="BUD-2024-001",
            budget_id="BUD-2024-001",
        ),
        RetrievedChunk(
            id=102,
            document_id=1,
            content="BUD-2024-001 :: PSD2-002 PSD2 open banking connectors — 160 h",
            sector="finance",
            project_year=2024,
            chunk_type="budget_component",
            distance=0.27,
            source_id="BUD-2024-001",
            budget_id="BUD-2024-001",
        ),
        RetrievedChunk(
            id=103,
            document_id=1,
            content="BUD-2024-001 :: TXN-003 Transaction ledger service — 140 h",
            sector="finance",
            project_year=2024,
            chunk_type="budget_component",
            distance=0.31,
            source_id="BUD-2024-001",
            budget_id="BUD-2024-001",
        ),
    ]


def _estimate() -> GroundedEstimate:
    """A grounded estimate with one planted dangling citation + one no-data line."""
    return GroundedEstimate(
        total_engineer_days=53,
        duration_weeks=11,
        modules=[
            EstimationModule(
                name="Authentication & SCA",
                tasks=[
                    EstimationTask(
                        name="OAuth 2.0 backend",
                        engineer_days=15,
                        grounded=True,
                        citations=[
                            SourceReference(
                                chunk_id="101",
                                document_id="BUD-2024-001",
                                evidence="AUTH-001 OAuth 2.0 authentication backend — 120 h",
                            )
                        ],
                    ),
                ],
            ),
            EstimationModule(
                name="PSD2 & Open Banking",
                tasks=[
                    EstimationTask(
                        name="Open banking connectors",
                        engineer_days=20,
                        grounded=True,
                        citations=[
                            SourceReference(
                                chunk_id="102",
                                document_id="BUD-2024-001",
                                evidence="PSD2-002 PSD2 open banking connectors — 160 h",
                            )
                        ],
                    ),
                ],
            ),
            EstimationModule(
                name="Ledger",
                tasks=[
                    # Planted DANGLING citation: chunk 999 was never retrieved.
                    EstimationTask(
                        name="Transaction ledger",
                        engineer_days=18,
                        grounded=True,
                        citations=[
                            SourceReference(
                                chunk_id="999",
                                document_id="BUD-2024-001",
                                evidence="TXN-003 Transaction ledger service — 140 h",
                            )
                        ],
                    ),
                ],
            ),
            EstimationModule(
                name="Reporting",
                tasks=[
                    # No sufficient source data: flagged, not back-filled with hours.
                    EstimationTask(name="Regulatory reporting", grounded=False),
                ],
            ),
        ],
        sources=[],
        assumptions=[
            Assumption(
                description="Regulatory reporting has no historical analog in the context.",
                impact="medium",
                rationale="No retrieved budget covers regulatory reporting.",
            )
        ],
        confidence="high",
        reasoning="Derived from the retrieved BUD-2024-001 components.",
    )


def _print_report(report: CitationReport) -> None:
    print("=== Citation verification report ===")
    print(
        f"lines: {report.total_lines}  grounded: {report.grounded_lines}  "
        f"dangling: {report.dangling_lines}  insufficient: {report.insufficient_lines}"
    )
    print(f"verified citations: {report.verified_citations}")
    print(f"dangling citations: {report.dangling_citations or '(none)'}")
    print()
    print(f"{'module':<24} {'line':<24} {'status':<13} cited -> dangling")
    print("-" * 88)
    for line in report.lines:
        print(
            f"{line.module:<24} {line.component:<24} {line.status:<13} "
            f"{line.cited_chunk_ids} -> {line.dangling_chunk_ids or '-'}"
        )


def main() -> int:
    chunks = _retrieved()
    estimate = _estimate()
    report = verify_citations_for_chunks(estimate, chunks)
    _print_report(report)

    # Acceptance checks (exit non-zero if any contract is violated).
    ok = (
        report.dangling_citations == ["999"]
        and report.dangling_lines == 1
        and report.grounded_lines == 2
        and report.insufficient_lines == 1
    )
    print()
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Ejecución esperada:

```bash
uv run python scripts/demo_verify_citations_s11.py
```

```text
=== Citation verification report ===
lines: 4  grounded: 2  dangling: 1  insufficient: 1
verified citations: 2
dangling citations: ['999']

module                   line                     status        cited -> dangling
----------------------------------------------------------------------------------------
Authentication & SCA     OAuth 2.0 backend        grounded      ['101'] -> -
PSD2 & Open Banking      Open banking connectors  grounded      ['102'] -> -
Ledger                   Transaction ledger       dangling      ['999'] -> ['999']
Reporting                Regulatory reporting     insufficient  [] -> -

ACCEPTANCE: PASS
```

### Paso 5 — Evaluación RAGAS

#### 5.1 — Golden set extendido

Crea `evals/golden_generation_s11.json` con el contenido de la sección **Datos para el ejercicio**.
No lo generes desde cero: parte de `evals/golden_retrieval.json`, copia Q1–Q5 y añade `transcript` y
`ground_truth`. Comprueba que las horas citadas en `ground_truth` existen en
`data/budgets_sample.json`.

#### 5.2 — Recogida de muestras

Crea `scripts/eval_ragas_s11.py`. Reproduce los pasos del pipeline completo
(`brief → retrieval → contexto → generación`) para poder capturar los `contexts` que RAGAS necesita.
No llama al endpoint HTTP: usa las mismas funciones que el orquestador.

> **Alcance del retrieval en la evaluación.** El collector fija
> `chunk_types=["budget_component"]` para que los `contexts` coincidan con el golden set budget-only
> de Q1–Q5. Reutiliza el mismo brief, el mismo retriever, el mismo ensamblado y el mismo generador
> que `estimate_from_transcript`; lo único que acota es la fuente, para que la tabla sea
> reproducible aunque el corpus multifuente esté ingerido. Si quieres medir también el camino
> multifuente, añade una segunda ejecución sin ese filtro y documenta que es otro baseline.

```python
#!/usr/bin/env python3
"""Collect RAGAS samples from the real Session 11 generation pipeline.

For each query in ``evals/golden_generation_s11.json`` this runs the REAL RAG
generation pipeline (brief -> embed -> retrieve -> assemble -> generate) and
records the four inputs RAGAS needs:

    question      -> the estimation request (golden ``query``)
    answer        -> the estimate the pipeline generated, rendered as text
    contexts      -> the retrieved chunks placed in the context block
    ground_truth  -> the expert reference estimate (golden ``ground_truth``)

The script only COLLECTS samples. Scoring happens in the isolated RAGAS venv
with ``scripts/score_ragas_s11.py`` (see the spec, Paso 0.1), because the
project's langchain stack and RAGAS's pinned langchain stack conflict.

Prerequisites (host run):
    * estimator/.env with OPEN_AI_KEY
    * the stack up and the budget corpus ingested
      (uv run python scripts/query_examples.py)
    * the dev environment synced: uv sync

Usage::

    uv run python scripts/eval_ragas_s11.py --collect-only /tmp/s11_samples.json
    uv run python scripts/eval_ragas_s11.py --limit 2 --collect-only /tmp/s11_samples.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.dependencies import (  # noqa: E402
    get_runtime_retrieval_config,
    get_semantic_retriever,
    get_token_encoder,
)
from app.generation.rag.brief_extractor import (  # noqa: E402
    compose_retrieval_text,
    extract_project_brief,
)
from app.generation.rag.context_builder import (  # noqa: E402
    TokenEncoder,
    assemble_context,
    fit_chunks_to_budget,
)
from app.generation.rag.grounded_estimator import generate_grounded_estimate  # noqa: E402
from app.generation.rag.grounding import verify_citations_for_chunks  # noqa: E402
from app.generation.rag.retriever import SemanticRetriever  # noqa: E402
from app.generation.rag.schemas import GroundedEstimate  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_generation_s11.json"

# Permissive threshold: we want the generator to see context for every query so
# RAGAS grades GENERATION, not the soft-fail gate.
GENERATION_THRESHOLD = 1.2


def render_estimate_text(estimate: GroundedEstimate) -> str:
    """Render a GroundedEstimate as the plain-text 'answer' RAGAS scores."""
    if estimate.confidence == "insufficient":
        return (
            "Insufficient context to produce an estimate. "
            f"{estimate.insufficient_context_explanation or ''}"
        ).strip()

    lines: list[str] = []
    if estimate.total_engineer_days is not None:
        lines.append(f"Total: {estimate.total_engineer_days} engineer-days.")
    if estimate.duration_weeks is not None:
        lines.append(f"Duration: {estimate.duration_weeks} weeks.")
    lines.append(f"Confidence: {estimate.confidence}.")
    for module in estimate.modules:
        lines.append(f"Module {module.name}:")
        for task in module.tasks:
            days = "n/a" if task.engineer_days is None else f"{task.engineer_days}d"
            tag = "grounded" if task.grounded else "no-data"
            citations = ", ".join(
                f"{ref.chunk_id}->{ref.document_id}" for ref in task.citations
            )
            suffix = f" sources=[{citations}]" if citations else ""
            lines.append(f"  - {task.name}: {days} [{tag}]{suffix}")
    if estimate.assumptions:
        lines.append(
            "Assumptions: " + "; ".join(assumption.description for assumption in estimate.assumptions)
        )
    lines.append(f"Reasoning: {estimate.reasoning}")
    return "\n".join(lines)


async def collect_sample(
    item: dict,
    retriever: SemanticRetriever,
    encoder: TokenEncoder,
) -> dict:
    """Run the pipeline for one golden item and return a RAGAS-ready sample."""
    transcript = item.get("transcript") or item["query"]
    brief = await extract_project_brief(transcript)
    retrieval_text = compose_retrieval_text(brief)

    runtime = get_runtime_retrieval_config()
    retrieval = await retriever.retrieve_chunks(
        query_text=retrieval_text,
        search_mode=runtime.effective_search_mode(),
        rerank=runtime.effective_rerank(),
        top_k=settings.rag_retrieval_top_k,
        recall_k=settings.retrieval_recall_top_k,
        rerank_top_n=settings.rerank_top_n,
        distance_threshold=GENERATION_THRESHOLD,
        rrf_k=settings.rrf_k,
        chunk_types=["budget_component"],
    )

    kept = fit_chunks_to_budget(
        retrieval.chunks,
        settings.rag_max_context_tokens,
        encoder,
    )
    context_block = assemble_context(kept)
    estimate = await generate_grounded_estimate(context_block, brief)
    report = verify_citations_for_chunks(estimate, kept)

    return {
        "id": item["id"],
        "question": item["query"],
        "answer": render_estimate_text(estimate),
        "contexts": [chunk.content for chunk in kept],
        "ground_truth": item["ground_truth"],
        "citation_report": report.model_dump(),
    }


async def _run(args: argparse.Namespace) -> int:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    retriever = get_semantic_retriever()
    if retriever is None:
        raise SystemExit("Embedding service is not configured (OPEN_AI_KEY?).")

    encoder = get_token_encoder()
    items = golden["queries"]
    if args.limit is not None:
        items = items[: args.limit]

    samples: list[dict] = []
    for item in items:
        print(f"Collecting {item['id']} ...")
        samples.append(await collect_sample(item, retriever, encoder))

    payload = {
        "judge_model": golden.get("judge_model", "gpt-4o-mini"),
        "embedding_model": golden.get("embedding_model", "text-embedding-3-small"),
        "samples": samples,
    }
    destination = args.collect_only
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {destination}")
    print(
        "Next: score in the isolated RAGAS venv:\n"
        f"  .venv-ragas/bin/python scripts/score_ragas_s11.py {destination} "
        "--out evals/ragas_baseline_s11.json"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-only",
        type=Path,
        required=True,
        metavar="PATH",
        help="Write the collected samples to PATH (JSON).",
    )
    parser.add_argument("--limit", type=int, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
```

#### 5.3 — Scoring con RAGAS

Crea `scripts/score_ragas_s11.py`. Importa **solo** RAGAS, `datasets` y `langchain-openai`, nunca
`app.*`, para poder correr en el venv aislado. Incluye los stubs de Vertex que RAGAS 0.4 necesita
aunque el juez sea OpenAI.

```python
#!/usr/bin/env python3
"""Standalone RAGAS scorer for pre-collected Session 11 samples.

This script imports ONLY ragas + langchain-openai + datasets — never the app —
so it runs in an isolated venv, decoupled from the project's newer langchain
stack (which conflicts with ragas's hard ``langchain_community`` Vertex import).
Collect the samples first with::

    uv run python scripts/eval_ragas_s11.py --collect-only /tmp/s11_samples.json

then score them here::

    .venv-ragas/bin/python scripts/score_ragas_s11.py /tmp/s11_samples.json \
        --out evals/ragas_baseline_s11.json

ragas 0.4.x unconditionally imports ``langchain_community.chat_models.vertexai``
at module load; recent ``langchain-community`` dropped that path. We never use
Vertex (the judge is OpenAI), so we register lightweight stub modules before
importing ragas to satisfy the import without pulling the Vertex SDK.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _install_vertex_shims() -> None:
    """Satisfy ragas's hard Vertex imports without the Vertex SDK (unused here)."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401

        return
    except Exception:
        pass

    class _Unavailable:
        def __init__(self, *args, **kwargs):  # pragma: no cover - defensive
            raise RuntimeError("Vertex AI is not available in this scorer venv.")

    chat_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    chat_mod.ChatVertexAI = _Unavailable
    sys.modules["langchain_community.chat_models.vertexai"] = chat_mod

    try:
        import langchain_community.llms as llms_mod  # type: ignore

        if not hasattr(llms_mod, "VertexAI"):
            llms_mod.VertexAI = _Unavailable
    except Exception:
        llms_mod = types.ModuleType("langchain_community.llms")
        llms_mod.VertexAI = _Unavailable
        sys.modules["langchain_community.llms"] = llms_mod


def _markdown_table(ids: list[str], rows: list[dict[str, float]]) -> str:
    header = "| query | " + " | ".join(METRIC_NAMES) + " |"
    separator = "|" + "---|" * (len(METRIC_NAMES) + 1)
    body = []
    for query_id, row in zip(ids, rows, strict=True):
        cells = " | ".join(f"{row[metric]:.3f}" for metric in METRIC_NAMES)
        body.append(f"| {query_id} | {cells} |")
    average = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRIC_NAMES}
    average_cells = " | ".join(f"{average[metric]:.3f}" for metric in METRIC_NAMES)
    body.append(f"| **average** | {average_cells} |")
    return "\n".join([header, separator, *body])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", help="JSON written by eval_ragas_s11.py --collect-only")
    parser.add_argument("--out", default=None, help="Write the scored report as JSON.")
    args = parser.parse_args()

    payload = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload
    judge_model = (
        payload.get("judge_model", "gpt-4o-mini") if isinstance(payload, dict) else "gpt-4o-mini"
    )
    embedding_model = (
        payload.get("embedding_model", "text-embedding-3-small")
        if isinstance(payload, dict)
        else "text-embedding-3-small"
    )

    _install_vertex_shims()

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(
        [
            {
                "question": sample["question"],
                "answer": sample["answer"],
                "contexts": sample["contexts"],
                "ground_truth": sample["ground_truth"],
            }
            for sample in samples
        ]
    )
    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))

    print(f"Scoring {len(samples)} samples (judge={judge_model}, embeddings={embedding_model})...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )

    frame = result.to_pandas()
    rows = [
        {
            metric: (float(frame.iloc[index][metric]) if metric in frame.columns else float("nan"))
            for metric in METRIC_NAMES
        }
        for index in range(len(samples))
    ]
    table = _markdown_table([sample["id"] for sample in samples], rows)

    print("\n=== RAGAS generation baseline ===")
    print(table)

    if args.out:
        report = {
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "table_markdown": table,
            "per_query": [
                {
                    "id": sample["id"],
                    **rows[index],
                    "citation_report": sample.get("citation_report"),
                }
                for index, sample in enumerate(samples)
            ],
        }
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Ejecución completa:

```bash
# 1) recoger muestras con el pipeline real (venv del proyecto)
uv run python scripts/eval_ragas_s11.py --collect-only /tmp/s11_samples.json

# 2) puntuar en el venv aislado. langchain-openai lee OPENAI_API_KEY; el
#    proyecto guarda la clave como OPEN_AI_KEY, así que expórtala con el nombre
#    que espera el scorer.
export OPENAI_API_KEY="${OPENAI_API_KEY:-$OPEN_AI_KEY}"
.venv-ragas/bin/python scripts/score_ragas_s11.py /tmp/s11_samples.json \
  --out evals/ragas_baseline_s11.json
```

> **Variables de entorno.** El pipeline del proyecto usa `OPEN_AI_KEY`; el scorer aislado usa
> `langchain-openai`, que lee `OPENAI_API_KEY`. Si tu `.env` define `OPEN_AI_KEY`, expórtala como
> `OPENAI_API_KEY` antes del paso 2 (o pasa la clave explícitamente al construir `ChatOpenAI` y
> `OpenAIEmbeddings`). No dupliques la clave en el repo.

#### 5.4 — Tabla de métricas

El scorer imprime y persiste una tabla con una fila por consulta y una fila de promedio:

| query | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| Q1 | — | — | — | — |
| Q2 | — | — | — | — |
| Q3 | — | — | — | — |
| Q4 | — | — | — | — |
| Q5 | — | — | — | — |
| **average** | — | — | — | — |

Sustituye los guiones por la salida real de tu ejecución. La tabla de referencia de la sección
**Datos para el ejercicio** no cuenta como entrega.

#### 5.5 — Documento de resultados y nota

Crea `docs/sesion-11-citacion-verificable-ragas.md` con:

1. Qué se cambió (schema, contexto, prompt, verificación).
2. El informe de citaciones de la demo (Paso 4) y el informe agregado de las cinco estimaciones
   reales (`citation_report` de cada muestra: líneas, grounded, dangling, insufficient, citas
   verificadas).
3. La tabla RAGAS real de 4 métricas × 5 consultas + promedio.
4. Una nota de **2–3 frases** con lo que más chirría de los números. Guía de lectura:
   - `faithfulness` baja con citación correcta suele significar que el generador descompone un
     componente histórico en varias subtareas y reparte horas; el juez no puede atribuir 1:1 cada
     cifra derivada a la cifra del `<source>`.
   - `context_recall` flojo en alguna consulta suele venir del desajuste de unidades
     (engineer-days frente a horas) o de una recuperación que trae el presupuesto correcto pero no
     los componentes que el `ground_truth` espera.
   - `answer_relevancy` ≈ 0 en casi todas es un artefacto de formato: la métrica regenera preguntas
     desde una tabla estructurada de módulos → tareas y las compara con un brief declarativo. No es
     una señal de que la estimación sea mala; es una señal de que la métrica no encaja con el
     formato.
   - `context_precision` perfecto con `context_recall` bajo es la firma de “traigo lo relevante,
     pero el juez no lo reconoce como la respuesta de referencia”.

### Paso 6 — Tests

Extiende los tests existentes. No hace falta Postgres ni red: los dobles ya están en el repo.

- `tests/test_grounding.py`:
  - `verify_citations` con todas las citas reales → `dangling_citations == []` y
    `grounded_lines == total_lines`.
  - Cita plantada `999` → `dangling_lines == 1`, `dangling_citations == ["999"]`.
  - Línea `grounded=False` → `insufficient_lines == 1` y no se rellena.
  - `document_id` incorrecto con chunk existente → `document_mismatches` y línea `dangling`.
  - `find_fabricated_citations` sigue devolviendo los ids legacy `[99, 100]`.
- `tests/test_rag_schemas.py`:
  - `EstimationTask(grounded=True)` sin `citations` → `ValidationError`.
  - `EstimationTask(grounded=False, engineer_days=10)` → `ValidationError`.
  - `EstimationTask` legacy con `sources=[1, 2]` y `engineer_days` sigue siendo válido.
  - `TaskHoursEstimate(has_match=True, grounded=True)` sin `sources` → `ValidationError` cuando
    `sources` se pasa explícitamente vacío.
  - `TaskHoursEstimate` legacy con `has_match=True` y sin campos nuevos sigue siendo válido.
- `tests/test_task_hours.py`:
  - Un match produce `grounded=True`, `sources` no vacías y `evidence` no vacía.
  - Un no-match produce `grounded=False`, `sources == []` y `estimated_hours is None`.
- `tests/test_rag_stages.py`:
  - `POST /estimate/stages/generate` devuelve `citation_report` con los contadores coherentes.
  - `POST /estimate/stages/structure` devuelve todas las líneas `insufficient` (sin fuentes).

Comandos:

```bash
uv run pytest tests/test_grounding.py tests/test_rag_schemas.py \
  tests/test_task_hours.py tests/test_rag_stages.py
uv run ruff check .
uv run ruff format --check .
uv run mypy app frontend
```

## Entregables

- Código del schema extendido (`SourceReference`, `EstimationTask.grounded/citations`,
  `TaskHoursEstimate.grounded/sources`, `CitationReport`).
- Prompts de atribución por línea (`rag_estimation/v3`, `rag_estimation/v4`,
  `rag_structure/v2`) y `<source document_id="...">`.
- `verify_citations` y su integración en el pipeline, en las etapas y en el consenso de horas.
- `scripts/demo_verify_citations_s11.py` con una cita colgante plantada y salida `ACCEPTANCE: PASS`.
- `evals/golden_generation_s11.json` con `ground_truth` por consulta.
- `scripts/eval_ragas_s11.py` y `scripts/score_ragas_s11.py`.
- `evals/ragas_baseline_s11.json` con las métricas por consulta y el `citation_report` de cada una.
- Tabla RAGAS real (4 métricas × 5 consultas + promedio) en
  `docs/sesion-11-citacion-verificable-ragas.md`.
- Nota breve (2–3 frases) sobre lo que más chirría de los números.
- Tests nuevos y actualizados.

## Criterios de aceptación

- [ ] Cada línea con `grounded=True` cita al menos una fuente real del contexto recuperado.
- [ ] La verificación detecta una citación colgante si se introduce a propósito (demo con `999`).
- [ ] Las líneas sin soporte se marcan como `insufficient`; no se rellenan con cifras inventadas.
- [ ] Una cita con `document_id` que no corresponde al presupuesto del chunk se reporta como
      colgante.
- [ ] El pipeline completo loguea `citations.verified` y `citations.unrepaired` con `request_id`.
- [ ] El endpoint por etapas devuelve `citation_report` y mantiene `fabricated_source_ids`.
- [ ] El consenso de horas devuelve `sources` verificables por tarea.
- [ ] RAGAS devuelve las cuatro métricas para las cinco consultas y el promedio.
- [ ] La tabla y la nota están en `docs/sesion-11-citacion-verificable-ragas.md`.
- [ ] `uv run pytest`, `ruff` y `mypy` en verde.

## Checklist de verificación

- [ ] `SourceReference` tiene `chunk_id`, `document_id` y `evidence`; `evidence` no está vacío.
- [ ] `EstimationTask.grounded=True` sin `citations` falla en validación.
- [ ] `EstimationTask.grounded=False` con `engineer_days` falla en validación.
- [ ] `EstimationTask.sources` legacy sigue funcionando para consumidores antiguos.
- [ ] `<source>` incluye `document_id` y el prompt lo pide explícitamente.
- [ ] Los nuevos prompts están en inglés y no mencionan `client.responses.parse`.
- [ ] `verify_citations` revisa líneas, citas legacy y citas globales.
- [ ] La demo imprime `dangling citations: ['999']` y `ACCEPTANCE: PASS`.
- [ ] El golden set tiene 5 consultas con `query`, `transcript`, `relevant_budget_ids` y
      `ground_truth`.
- [ ] El scorer produce una tabla con 5 filas y promedio.
- [ ] `evals/ragas_baseline_s11.json` existe y contiene `citation_report` por consulta.
- [ ] `docs/sesion-11-citacion-verificable-ragas.md` contiene la tabla real, no la de referencia.
- [ ] `git status` no incluye `.venv-ragas/` ni artefactos temporales.
- [ ] La rama es `feature/pre-session-11` y el PR apunta a `main`.

## Documentación de referencia

- Spec de la Sesión 10 (retrieval híbrido y reranking):
  [`sesion-10-busqueda-hibrida-reranking.md`](./sesion-10-busqueda-hibrida-reranking.md)
- Documento de la Sesión 10 multifuente:
  [`../docs/sesion-10-multifuente.md`](../docs/sesion-10-multifuente.md)
- Spec del pre-work de la Sesión 09:
  [`sesion-9-rag-end-to-end-prework.md`](./sesion-9-rag-end-to-end-prework.md)
- RAGAS — métricas y evaluación:
  https://docs.ragas.io/
- OpenAI — embeddings (`text-embedding-3-small`):
  https://platform.openai.com/docs/guides/embeddings
- Pydantic v2 — validadores de modelo:
  https://docs.pydantic.dev/latest/concepts/validators/
- Jinja2 — `StrictUndefined` y plantillas:
  https://jinja.palletsprojects.com/

## Nota

**Qué es nuestro (y por qué).** El repo de referencia resuelve la sesión con `TaskItem`,
`Estimate`, `context_assembler.py` y `validation.py`. Este repo ya tiene `EstimationTask`,
`GroundedEstimate`, `context_builder.py` y `grounding.py`, y además un camino S10 de estructura sin
horas con consenso por tarea que el repo de referencia no tiene. La spec adapta la idea, no los
nombres: la cita por línea vive en las dos etapas y la verificación se integra donde ya se validan
las citas. No se crea un modelo paralelo ni se rompe el contrato HTTP.

**Lo que NO hay que hacer.** No cambies el retrieval para “mejorar” RAGAS. No metas RAGAS en
`pyproject.toml`. No inventes horas para las líneas sin soporte. No interpretes `answer_relevancy`
bajo como un fallo del generador sin mirar el formato de la respuesta. No copies la tabla de
referencia en el documento de resultados: ejecuta el pipeline y publica tus números.

**La trampa de la sesión.** Una cita que existe pero apunta al presupuesto equivocado es tan mala
como una cita inventada. Por eso `verify_citations` comprueba `chunk_id` **y** `document_id`, y por
eso el informe distingue `dangling` de `insufficient`: una línea sin datos no es un fallo de
citación, es una línea honesta que debe llegar a la persona que valida la estimación.
