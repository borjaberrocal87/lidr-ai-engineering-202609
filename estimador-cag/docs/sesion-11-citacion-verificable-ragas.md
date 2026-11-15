# Sesión 11 — Citación verificable y baseline RAGAS

Esta nota recoge la evidencia del pre-work de la Sesión 11 sobre el repo actual:

1. **Citación verificable a nivel de línea** y detección de citaciones colgantes.
2. **Baseline RAGAS** de generación sobre las cinco consultas del golden set extendido.

Todo el código nuevo está en inglés; esta nota y el `ground_truth` pueden estar en español.

## Qué se cambió (Parte 1)

- **Schema** (`app/generation/rag/schemas.py`):
  - `SourceReference{chunk_id, document_id, evidence}` — cita verificable de una línea.
  - `EstimationTask` añade `grounded` y `citations`; conserva `sources: list[int]` como campo legacy.
  - `TaskHoursEstimate` añade `grounded` y `sources`; conserva `has_match`.
  - `CitationReport` / `LineCitation` — salida de la verificación.
- **Contexto** (`app/generation/rag/context_builder.py`): cada `<source>` expone `document_id`
  (el `budget_id` del presupuesto o el `source_id` de la fuente).
- **Prompts**: `rag_estimation/v3` (con horas), `rag_estimation/v4` (estructura con contexto) y
  `rag_structure/v2` (estructura sin fuentes). La atribución por línea pide `chunk_id`,
  `document_id` y `evidence` **verbatim**; las líneas sin soporte se marcan `grounded=false`.
- **Verificación** (`app/generation/rag/grounding.py`): `verify_citations` comprueba `chunk_id` y
  `document_id` contra los chunks realmente entregados. El pipeline completo reintenta una vez y
  loguea `citations.verified` / `citations.unrepaired` con `request_id`; el endpoint por etapas
  devuelve `citation_report`; el consenso de horas devuelve `sources` por tarea.

## Informe de citaciones de la demo (offline)

Salida real de `uv run python scripts/demo_verify_citations_s11.py`, que planta una cita colgante
(`999`) y una línea sin datos:

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

- Las líneas `grounded=True` citan chunks reales del contexto (`101`, `102`).
- La cita colgante `999` se detecta y se reporta como `dangling`.
- La línea sin soporte se marca `insufficient`; no se rellena con horas inventadas.

## Informe de citaciones de las estimaciones reales

Generado por `scripts/eval_ragas_s11.py` sobre las cinco consultas y verificado con
`verify_citations_for_chunks`:

| query | líneas | grounded | dangling | insufficient | citas verificadas |
|---|---|---|---|---|---|
| Q1 | 12 | 4 | 0 | 8 | 4 |
| Q2 | 15 | 15 | 0 | 0 | 15 |
| Q3 | 13 | 13 | 0 | 0 | 13 |
| Q4 | 9 | 9 | 0 | 0 | 9 |
| Q5 | 20 | 20 | 0 | 0 | 20 |
| **total** | **69** | **61** | **0** | **8** | **61** |

**Citaciones colgantes: 0/69 líneas.** Las 8 líneas sin soporte (todas en Q1) se marcan
`insufficient`, no se rellenan con cifras. La recuperación de Q1 trajo solo 10 chunks y el modelo
solo encontró evidencia suficiente para 4 de las 12 tareas que propuso.

## Baseline RAGAS (Parte 2)

- **Golden set**: `evals/golden_generation_s11.json` — Q1–Q5 de la Sesión 10 con `transcript` y
  `ground_truth` (engineer-days, ~horas/8).
- **Pipeline evaluado**: brief → retrieval budget-only (`chunk_types=["budget_component"]`) →
  ensamblado → generación v3. Se usa el mismo camino que `estimate_from_transcript`; el filtro de
  fuente mantiene la tabla reproducible.
- **Juez**: `gpt-4o-mini`; **embeddings**: `text-embedding-3-small`.
- **Métricas**: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
- **Datos completos**: `evals/ragas_baseline_s11.json` (incluye el `citation_report` por consulta).

| query | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| Q1 | 0.435 | 0.000 | 1.000 | 0.000 |
| Q2 | 0.000 | 0.068 | 1.000 | 0.000 |
| Q3 | 0.100 | 0.498 | 1.000 | 0.000 |
| Q4 | 0.294 | 0.000 | 1.000 | 0.667 |
| Q5 | 0.086 | 0.107 | 0.957 | 0.571 |
| **average** | 0.183 | 0.135 | 0.991 | 0.248 |

### Nota de hallazgos

Lo que más chirría es el contraste entre una **precisión de contexto casi perfecta (0.991)** y un
**recall de contexto bajo (0.248)**, con `faithfulness` media de 0.183: la recuperación trae
presupuestos relevantes, pero el `ground_truth` está en engineer-days, referencia componentes
concretos y el juez no consigue alinear esas cifras con el contexto. Q1 es el caso extremo
(8/12 líneas `insufficient`), mientras Q5 es el mejor (0.571 de recall y 0.957 de precisión).
`answer_relevancy` (0.135) está dominada por un artefacto de formato: la "respuesta" es una tabla
estructurada de módulos → tareas y la métrica regenera preguntas desde ella; no mide bien este
formato, así que no debe leerse como un juicio de calidad de la estimación.

## Cómo reproducirlo

```bash
# 1) Stack arriba y corpus budget ingerido
docker compose up -d --build
docker compose run --rm api python scripts/query_examples.py

# 2) Recoger muestras con el pipeline real
uv run python scripts/eval_ragas_s11.py --collect-only /tmp/s11_samples.json

# 3) Puntuar en el venv aislado de RAGAS
uv venv .venv-ragas --python 3.12
uv pip install --python .venv-ragas/bin/python "ragas>=0.4,<0.5" datasets langchain-openai
export OPENAI_API_KEY="$(grep -E '^OPEN_AI_KEY=' .env | cut -d= -f2- | tr -d '\"')"
.venv-ragas/bin/python scripts/score_ragas_s11.py /tmp/s11_samples.json \
  --out evals/ragas_baseline_s11.json

# 4) Demo offline de citación colgante
uv run python scripts/demo_verify_citations_s11.py
```

> `OPEN_AI_KEY` es la variable del proyecto; `langchain-openai` lee `OPENAI_API_KEY`. Por eso el
> paso 3 la exporta con el nombre que espera el scorer. No se duplica la clave en el repo.

## Limitaciones conocidas

- La tabla RAGAS se midió con la configuración de retrieval por defecto (`vector`, sin reranking).
  Cambiar `RETRIEVAL_SEARCH_MODE` o `RERANKER_ENABLED` produce otro baseline; hay que documentarlo.
- `answer_relevancy` no encaja con una respuesta estructurada de módulos → tareas. Se mantiene
  porque es una de las cuatro métricas pedidas, pero no es la señal principal de calidad.
- El `ground_truth` está en engineer-days y el corpus en horas; es la causa más probable del
  `context_recall` bajo. Una siguiente iteración puede normalizar unidades o anotar el golden set
  con las horas históricas exactas.
