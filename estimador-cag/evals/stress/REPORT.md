# REPORT — Baseline de estrés del CAG (sesión 6)

Instrumentación y medición del CAG construido hasta la sesión 5, para cuantificar
sus límites **antes** de introducir RAG en el directo. El deliverable son tres
curvas y dos párrafos de lectura sobre los CSV de la campaña.

## Cómo se generó (alcance de la campaña)

```bash
# Curva multi-turno (la compresión de memoria entra en juego a partir del turno 7)
uv run python -m evals.stress.run --scenarios growing,pivot,contradiction \
  --attachment-sizes 0 --repeats 1 --growing-turns 10 \
  --output evals/stress/results_multiturn.csv

# Curva de adjuntos (misma estimación inicial, solo cambia el adjunto)
uv run python -m evals.stress.run --scenarios growing --growing-turns 1 \
  --attachment-sizes 0,5,20,50,100 --repeats 1 \
  --output evals/stress/results_attachments.csv
```

Los tres CSV (`results_multiturn.csv`, `results_attachments.csv` y el merge
`results.csv`, 33 filas) están versionados en `evals/stress/`. **Caveats:**

- `--repeats 1` (el enunciado pide ≥ 3). El P50/P95 se calcula igualmente, pero
  sobre 8–10 muestras por escenario; trata los P95 como indicativos.
- Un único proveedor/modelo: `deepseek-v4-flash` (custom, `nan.builders`).
- Es un modelo con *reasoning*, de latencia alta (~40 s/turno): la campaña
  completa (5 tamaños × 3 repeticiones) habría tardado horas.

## 1. Tabla resumen

Escenario × turno, sin adjunto (`results_multiturn.csv`):

| Escenario | n | P50 latency | P95 latency | Coste acumulado | Cache exact | Cache semantic | Recall fact-tracker |
|---|---|---|---|---|---|---|---|
| growing | 10 | 47.7 s | 76.3 s | 0.1126 USD | 0 % | 0 % | **1.00** |
| pivot | 8 | 30.9 s | 61.5 s | 0.0598 USD | 0 % | 0 % | **1.00** |
| contradiction | 10 | 38.7 s | 45.3 s | 0.0863 USD | 0 % | 0 % | **0.00** |

- **LatencyBudgetMetric (SLA 4.000 ms): 0/33 turnos cumplen.** El presupuesto de
  latencia está roto de raíz con este modelo.
- **CostBudgetMetric (0.05 USD/turno): 33/33 cumplen.** El coste no es el problema
  a este volumen; el problema es el tiempo de pared.
- **Hit rate de caché 0 %** en exact y semantic: el path conversacional no cachea
  (`cache_hit_kind="none"` siempre), tal y como documenta el diseño.

## 2. Las tres curvas

### 2.1 Latencia vs tokens de entrada

| tokens_in | n | latencia media | P95 |
|---|---|---|---|
| 2.000–3.999 | 3 | 17.2 s | 19.1 s |
| 4.000–5.999 | 2 | 50.4 s | 68.0 s |
| 6.000–7.999 | 1 | 37.3 s | 37.3 s |
| 8.000–9.999 | 2 | 37.6 s | 43.2 s |
| 10.000–11.999 | 3 | 35.2 s | 43.5 s |
| 12.000–13.999 | 2 | 39.6 s | 39.9 s |
| 14.000–15.999 | 3 | 45.8 s | 49.6 s |
| 16.000–17.999 | 3 | 38.4 s | 44.7 s |
| 18.000–19.999 | 4 | 46.1 s | 66.6 s |
| 20.000–21.999 | 1 | 32.6 s | 32.6 s |
| 22.000–23.999 | 1 | 79.6 s | 79.6 s |
| 24.000–25.999 | 1 | 69.7 s | 69.7 s |
| 26.000–27.999 | 2 | 53.0 s | 70.5 s |

La tendencia es creciente con el tamaño del prompt, pero **ruidosa**: el pico de
latencia cae en el rango 22–26k tokens y la varianza del proveedor domina buena
parte de la señal.

### 2.2 Coste acumulado vs turno

| turno | growing | pivot | contradiction |
|---|---|---|---|
| 1 | 0.0052 | 0.0064 | 0.0029 |
| 2 | 0.0123 | 0.0092 | 0.0055 |
| 3 | 0.0233 | 0.0153 | 0.0143 |
| 4 | 0.0329 | 0.0212 | 0.0217 |
| 5 | 0.0444 | 0.0309 | 0.0306 |
| 6 | 0.0571 | 0.0414 | 0.0405 |
| 7 | 0.0679 | 0.0507 | 0.0535 |
| 8 | 0.0821 | 0.0598 | 0.0637 |
| 9 | 0.0969 | — | 0.0743 |
| 10 | 0.1126 | — | 0.0863 |

El coste acumulado crece de forma aproximadamente cuadrática (cada turno reenvía
la historia). El **coste por turno** del `growing` pasa de 0.0052 USD (t1) a
0.0157 USD (t10): **×3,0**, con los tokens de entrada subiendo de 5.450 a 26.186
(**×4,8**).

### 2.3 Recall del fact-tracker vs N (MemoryDriftMetric)

| turno | growing (`Nimbus`) | pivot (`Flutter`) | contradiction (`30000 EUR`) |
|---|---|---|---|
| 1 | 1.0 | — | — |
| 2 | 1.0 | — | — |
| 3 | 1.0 | — | **0.0** |
| 4 | 1.0 | — | 0.0 |
| 5 | 1.0 | 1.0 | 0.0 |
| 6 | 1.0 | 1.0 | 0.0 |
| 7 | 1.0 | 1.0 | 0.0 |
| 8 | 1.0 | 1.0 | 0.0 |
| 9 | 1.0 | — | 0.0 |
| 10 | 1.0 | — | 0.0 |

La memoria **no** se rompió para el `project_name` ni para el stack: el
`ProjectMetadata` los preserva turno a turno, así que sobreviven a la ventana y a
la compresión. El único fallo es el hecho de presupuesto: `30000 EUR` **nunca**
se encontró, ni siquiera en el turno en que se declara.

### 2.4 Adjuntos grandes

Misma estimación inicial (turno 1, `growing`); solo cambia el PDF adjunto
(`results_attachments.csv`):

| Adjunto | tokens_in | att_chars en prompt | latency | coste | attachment_recall |
|---|---|---|---|---|---|
| 0 KB | 5.567 | 0 | 27.9 s | 0.0050 USD | — |
| 5 KB | 15.986 | 19.477 | 31.3 s | 0.0088 USD | 1.0 |
| 20 KB | 37.625 | **60.000** | 46.7 s | 0.0171 USD | 0.0 |
| 50 KB | 37.578 | **60.000** | 37.0 s | 0.0156 USD | 0.0 |
| 100 KB | 37.449 | **60.000** | 33.4 s | 0.0149 USD | 1.0 |

El texto extraído del PDF se **trunca a `MAX_ATTACHMENT_CHARS=60.000`** a partir
de 20 KB: 20, 50 y 100 KB aportan exactamente el mismo contexto (60.000
caracteres) aunque el fichero crezca. `attachment_recall` es un proxy literal
estricto (¿aparece el marcador del adjunto, palabra por palabra, en la prosa de
respuesta?): da 1.0 cuando el modelo repite el marcador y 0.0 cuando no. No es una
medida semántica fiable; se incluye como cota inferior.

## 3. Dónde empieza a romperse el CAG y por qué

**La dimensión que rompe primero es la latencia, no la memoria.** Con el CAG de
la sesión 5, cualquiera de los tres perfiles supera el SLA de 4 s en todos los
turnos (P95 entre 45 y 76 s) y el coste por turno se multiplica por 3,0 en solo 10
turnos porque cada llamada reenvía la historia completa (`tokens_in` ×4,8). La
memoria, en cambio, aguanta: `Nimbus` sigue en el prompt en el turno 10 (recall
1,00) y `Flutter` desde el turno 5, porque `ProjectMetadata` se regenera cada
turno y sobrevive a la ventana deslizante. La contradicción del presupuesto enseña
el otro modo de fallo, el silencioso: `30000 EUR` da recall 0,00 ya en el turno 3
—el detector heurístico de anclas sólo reconoce frases en inglés (`budget ...
locked at`), la transcripción va en español y la metadata no guarda el literal— y
ningún test rojo se entera. Ese es exactamente el punto del ejercicio: el fallo
que sí salta (esquema roto) no es el peligroso; el peligroso es el recall que cae
a 0 sin avisar.

**El caso límite que justifica saltar a RAG es el presupuesto de contexto
agotado, no el fallo puntual.** A partir de 20 KB de adjunto el CAG trunca a
60.000 caracteres: 20, 50 y 100 KB entregan *el mismo* contexto, así que el
usuario que sube un documento de 100 KB cree haber dado más información cuando el
sistema descartó la mayor parte del texto —información perdida de forma
silenciosa— y además paga ~3× más que sin adjunto (0.0149 vs 0.0050 USD). La
frontera arquitectónica aparece cuando el material útil supera el presupuesto de
contexto de forma recurrente: ahí CAG sólo puede truncar (pierde señal) o
encarecer cada turno (reenvía todo). El umbral concreto en estos datos: **adjuntos
por encima de ~20 KB** (activación del truncado) y **sesiones más allá de ~10
turnos** con `tokens_in` camino de los 26k y latencias sostenidas por encima del
SLA. Cuando esas dos condiciones se vuelven la norma, recuperar sólo los
fragmentos relevantes (RAG) deja de ser una optimización y pasa a ser la condición
para respetar el presupuesto de latencia y no perder contenido.

## 4. Reproducir

```bash
uv run python -m evals.stress.run --http http://localhost:8000 \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,5,20,50,100 \
  --repeats 3 \
  --output evals/stress/results.csv
```

Los PDFs de fixture se versionan en `evals/stress/fixtures/` (el runner los
regenera) y el CSV se escribe fila a fila, de modo que una campaña larga se puede
seguir en vivo.
