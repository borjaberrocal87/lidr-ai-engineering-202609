# Evals y golden dataset

> Cómo medir que el estimator no ha regresado, con una suite que corre sin LLM
> gracias a métricas deterministas.

## Qué es y para qué sirve

`evals/` es un **harness de evaluación**: coge un conjunto fijo de casos dorados
(`golden_dataset.json`), los lanza contra la app FastAPI real (en proceso o por
HTTP) y puntúa cada respuesta con métricas deterministas. Sirve para:

- Detectar regresiones (el modelo se sale de presupuesto, deja de mencionar una
  tecnología, rompe el contrato del esquema).
- Comparar los modos **actor** y **acb** sobre el mismo dataset.
- Servir de red de seguridad complementaria a los tests unitarios: los tests
  mockean el LLM; los evals lo ejecutan de verdad.

## Estructura

```
evals/
├── __init__.py            # re-exports públicos
├── dataset.py             # GoldenCase + load_dataset()
├── metrics.py             # métricas deterministas
├── run.py                 # CLI (python -m evals.run)
└── golden_dataset.json    # 16 casos preinstalados
```

## El golden dataset

Cada caso lo describe el modelo `GoldenCase` (`evals/dataset.py`):

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `id` | str | Identificador único del caso. |
| `transcript` | str | Transcripción de entrada (20–80.000 chars). |
| `project_type` | enum | `mobile_app` \| `web_saas` \| `internal_tool` \| `data_pipeline`. |
| `detail_level` | enum | `summary` \| `medium` \| `detailed`. |
| `output_format` | enum | `phases_table` \| `line_items` \| `narrative`. |
| `tier` | enum | `auto` \| `executive` \| `pm` \| `developer` \| `default` (`auto` = deja que el resolver decida). |
| `expected_out_of_scope` | bool | El caso *debe* producir el sobre fuera-de-alcance (cost=0, duration=1). |
| `expected_in_summary` | list[str] | Términos que deben aparecer en el texto. |
| `expected_technologies_any_of` | list[str] | Al menos una de estas tecnologías debe mencionarse. |
| `expected_phase_count_range` | [lo, hi] | Rango aceptable de fases. |
| `expected_cost_range_eur` | [lo, hi] | Rango aceptable de coste total. |
| `expected_duration_weeks_range` | [lo, hi] | Rango aceptable de duración. |
| `expected_tier` | enum | Tier esperado (documental; se conserva para futuras métricas de tier). |

Los valores son **rangos laxos**, no igualdad exacta: un LLM tiene latitud
legítima al dimensionar fases. El objetivo es cazar regresiones obvias, no fijar
cada dígito.

El dataset preinstalado mezcla tipos de proyecto y "sabores" (NDA, HIPAA, GDPR,
equipo pequeño, microservicios, fintech) más dos adversarios fuera de alcance
(un transcript vago y un intento de prompt injection).

## Las métricas (`evals/metrics.py`)

Todas son deterministas (sin LLM) y devuelven `MetricResult(name, score, passed, details)`:

- **`schema_adherence`** — la suma de `phase.cost_eur` cuadra con `total_cost_eur`;
  si `confidence_pct < 30`, el summary empieza por `"Out of scope:"`; y el número
  de fases cae dentro del rango esperado. Existe para detectar si un validador de
  negocio se pierde en una refactorización.
- **`cost_bounds`** — el coste/duración caen en el rango esperado. Para casos
  fuera de alcance, exige exactamente el sobre placeholder (cost=0, duration=1).
- **`content_recall`** — recall ligero: los términos de `expected_in_summary`
  aparecen y al menos una tecnología esperada se menciona. Se salta en casos
  fuera de alcance.

`run_all_metrics(case, result)` ejecuta las tres y devuelve la lista.

> Opcional: `metrics.py` está pensado para poder cablear un juez LLM (DeepEval
> `GEval`) como segunda opinión cualitativa, pero **por defecto no lo usa**: la
> gracia de tenerlo en el árbol es que la suite corre sin coste ni claves.

## Cómo ejecutarlo

```bash
# Modo actor, todos los casos, contra la app en proceso (TestClient)
uv run python -m evals.run --mode actor

# Modo Actor-Critic-Boss, primeros 5 casos, guardando el informe
uv run python -m evals.run --mode acb --limit 5 --output /tmp/acb.json

# Contra un servidor ya arrancado
uv run python -m evals.run --mode actor --http http://localhost:8000
```

Salida (por caso y por métrica) + recuento final:

```
Running 16 cases against mode=actor
case-001-web-saas-crm         | 1234 ms | schema_adherence=PASS(1.00) | cost_bounds=PASS(1.00) | content_recall=PASS(1.00)
...
Summary
  16/16 cases passing all metrics
  schema_adherence: 16/16
  cost_bounds: 16/16
  content_recall: 16/16
```

El runner usa un `SessionStore` aislado (override de dependencia) para no
contaminar sesiones del servidor de desarrollo. Devuelve código de salida 1 si
algún caso falla alguna métrica, de modo que pueda usarse en CI.

> Requiere un proveedor LLM configurado (los evals llaman al modelo de verdad).
> Los tests unitarios, en cambio, no tocan la red.

## Cómo añadir casos

1. Añade un objeto al array de `evals/golden_dataset.json` con un `id` único.
2. Rellena los rangos esperados de coste/duración/fases con criterio (generosos:
   el objetivo es cazar regresiones, no clavar cifras).
3. Si el caso debe ser rechazado por el guardrail de entrada, marca
   `expected_out_of_scope: true`: el runner mapea el 400 del guardrail al sobre
   fuera-de-alcance y lo puntúa como pase.
4. Ejecuta `uv run python -m evals.run --limit <n>` para validar tu caso.

## Tests deterministas vs evals

| | Tests (`pytest`) | Evals (`python -m evals.run`) |
| --- | --- | --- |
| LLM | Mockeado (`FakeWrapper`) | Real |
| Coste | Ninguno | Tokens reales |
| Velocidad | Milisegundos | Segundos por caso |
| Objetivo | Correctitud del código | Calidad/robustez de la salida del modelo |
| CI | Sí | Manual / nocturno |

El test `tests/test_evals_dataset.py` valida la **carga** del dataset y el
**contrato** de las métricas sin llamar al LLM.
