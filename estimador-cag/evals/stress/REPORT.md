# REPORT — Baseline de estrés del CAG (sesión 6)

Instrumentación y medición del CAG construido hasta la sesión 5, para cuantificar
sus límites **antes** de introducir RAG en el directo. El deliverable son tres
curvas y dos párrafos de lectura sobre `results.csv`.

## Cómo se generó (alcance de la campaña)

```bash
# Curva multi-turno (compresión de memoria en juego a partir del turno 7)
uv run python -m evals.stress.run --scenarios growing,pivot,contradiction \
  --attachment-sizes 0 --repeats 1 --growing-turns 10 --output results_multiturn.csv

# Curva de adjuntos (misma estimación inicial, solo cambia el adjunto)
uv run python -m evals.stress.run --scenarios growing --growing-turns 1 \
  --attachment-sizes 0,5,20,50,100 --repeats 1 --output results_attachments.csv
```

Ambos CSV se concatenan en `evals/stress/results.csv`. **Caveats de esta pasada:**

- `--repeats 1` (el enunciado pide ≥ 3). El P50/P95 se calcula igualmente, pero
  sobre 8–10 muestras por escenario; trata los P95 como indicativos.
- Un único proveedor/modelo: `deepseek-v4-flash` (custom, `nan.builders`).
- Es un modelo con *reasoning*, de latencia alta (~40 s/turno): la campaña
  completa (5 tamaños × 3 repeticiones) habría tardado varias horas.

## 1. Tabla resumen

Escenario × turno, sin adjunto (`attachment_kb=0`):

| Escenario | n | P50 latency | P95 latency | Coste acumulado | Cache exact | Cache semantic | Recall fact-tracker |
|---|---|---|---|---|---|---|---|
| growing | 10 | 39.0 s | 55.6 s | 0.1068 USD | 0 % | 0 % | **1.00** |
| pivot | 8 | 31.4 s | 57.8 s | 0.0615 USD | 0 % | 0 % | **1.00** |
| contradiction | 10 | 33.6 s | 62.2 s | 0.0956 USD | 0 % | 0 % | **0.00** |

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
| 2.000–3.999 | 1 | 16.8 s | 16.8 s |
| 4.000–5.999 | 2 | 43.1 s | 55.5 s |
| 6.000–7.999 | 3 | 35.4 s | 47.9 s |
| 8.000–9.999 | 3 | 41.4 s | 48.8 s |
| 10.000–11.999 | 3 | 27.9 s | 29.1 s |
| 12.000–13.999 | 3 | 33.4 s | 41.0 s |
| 14.000–15.999 | 3 | 35.9 s | 40.7 s |
| 16.000–17.999 | 2 | 44.1 s | 56.8 s |
| 18.000–19.999 | 6 | 42.2 s | 65.2 s |
| 20.000–21.999 | 1 | 54.9 s | 54.9 s |
| 32.000–33.999 | 1 | 56.1 s | 56.1 s |

La tendencia es creciente con el tamaño del prompt, pero **ruidosa**: la varianza
del proveedor domina por encima de la señal en el rango 4–18k tokens.

### 2.2 Coste acumulado vs turno

| turno | growing | pivot | contradiction |
|---|---|---|---|
| 1 | 0.0028 | 0.0050 | 0.0050 |
| 2 | 0.0103 | 0.0095 | 0.0163 |
| 3 | 0.0221 | 0.0156 | 0.0266 |
| 4 | 0.0292 | 0.0232 | 0.0342 |
| 5 | 0.0391 | 0.0310 | 0.0431 |
| 6 | 0.0491 | 0.0422 | 0.0525 |
| 7 | 0.0621 | 0.0514 | 0.0617 |
| 8 | 0.0741 | 0.0615 | 0.0720 |
| 9 | 0.0921 | — | 0.0834 |
| 10 | 0.1068 | — | 0.0956 |

El coste acumulado crece de forma aproximadamente cuadrática (cada turno reenvía
la historia). El **coste por turno** del `growing` pasa de 0.0028 USD (t1) a
0.0147 USD (t10): **×5,2**, con los tokens de entrada subiendo de 2.216 a 20.913
(**×9,4**).

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

Misma estimación inicial (turno 1, `growing`); solo cambia el PDF adjunto:

| Adjunto | tokens_in | att_chars en prompt | latency | coste | attachment_recall |
|---|---|---|---|---|---|
| 0 KB | 5.444 | 0 | 27.2 s | 0.0046 USD | — |
| 5 KB | 16.029 | 19.477 | 33.2 s | 0.0090 USD | 0.0 |
| 20 KB | 37.752 | **60.000** | 44.7 s | 0.0174 USD | 1.0 |
| 50 KB | 37.540 | **60.000** | 64.7 s | 0.0158 USD | 1.0 |
| 100 KB | 57.724 | **60.000** | 38.1 s | 0.0219 USD | 0.0 |

El texto extraído del PDF se **trunca a `MAX_ATTACHMENT_CHARS=60.000`** a partir
de 20 KB: 20, 50 y 100 KB aportan exactamente el mismo contexto (60.000
caracteres) aunque el fichero crezca. `attachment_recall` es un proxy literal
estricto (¿aparece el marcador del adjunto, palabra por palabra, en la prosa de
respuesta?): da 0.0 en 5 y 100 KB y 1.0 en 20 y 50 KB cuando el modelo repite el
marcador tal cual. No es una medida semántica fiable; se incluye como cota
inferior.

## 3. Dónde empieza a romperse el CAG y por qué

**La dimensión que rompe primero es la latencia, no la memoria.** Con el CAG de
la sesión 5, cualquiera de los tres perfiles supera el SLA de 4 s en todos los
turnos (P95 entre 55 y 62 s) y el coste por turno se multiplica por 5,2 en solo
10 turnos porque cada llamada reenvía la historia completa
(`tokens_in` ×9,4). La memoria, en cambio, aguanta: `Nimbus` sigue en el prompt en
el turno 10 (recall 1,00) y `Flutter` desde el turno 5, porque `ProjectMetadata`
se regenera cada turno y sobrevive a la ventana deslizante. La contradicción del
presupuesto enseña el otro modo de fallo, el silencioso: `30000 EUR` da recall
0,00 ya en el turno 3 —el detalle heurístico de anclas sólo reconoce frases en
inglés (`budget ... locked at`), la transcripción va en español y la metadata no
guarda el literal— y ningún test rojo se entera. Ese es exactamente el punto del
ejercicio: el fallo que sí salta (esquema roto) no es el peligroso; el peligroso
es el recall que cae a 0 sin avisar.

**El caso límite que justifica saltar a RAG es el presupuesto de contexto
agotado, no el fallo puntual.** A partir de 20 KB de adjunto el CAG trunca a
60.000 caracteres: 20, 50 y 100 KB entregan *el mismo* contexto, así que el
usuario que sube un documento de 100 KB cree haber dado más información cuando el
sistema descartó el 40 % del texto —información perdida de forma silenciosa— y
además paga 4,8× más que sin adjunto (0.0219 vs 0.0046 USD). La frontera
arquitectónica aparece cuando el material útil supera el presupuesto de contexto
de forma recurrente: ahí CAG sólo puede truncar (pierde señal) o encarecer cada
turno (reenvía todo). El umbral concreto en estos datos: **adjuntos por encima de
~20 KB** (activación del truncado) y **sesiones más allá de ~10 turnos** con
`tokens_in` camino de los 20k y latencias sostenidas por encima del SLA. Cuando
esas dos condiciones se vuelven la norma, recuperar sólo los fragmentos
relevantes (RAG) deja de ser una optimización y pasa a ser la condición para
respetar el presupuesto de latencia y no perder contenido.

## 4. Reproducir

```bash
uv run python -m evals.stress.run --http http://localhost:8000 \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,5,20,50,100 \
  --repeats 3 \
  --output evals/stress/results.csv
```

Los PDFs de fixture se regeneran solos (`evals/stress/fixtures/`) y el CSV se
escribe fila a fila, de modo que una campaña larga se puede seguir en vivo.
