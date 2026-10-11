# Patrón Actor-Critic-Boss (ACB)

> Cómo el endpoint `POST /api/v1/sessions/{session_id}/estimate-acb` produce una
> estimación revisada, con traza de auditoría, sin renunciar a la resiliencia.

## Qué problema resuelve

Un único LLM que estima "de una pasada" comete errores que nadie audita: suma mal
las fases, inventa tecnologías, o rompe el contrato de audiencia. El patrón ACB
añade un **revisor independiente** (Critic) y un **orquestador sin LLM** (Boss)
que decide si la respuesta es aceptable, si merece otra pasada con feedback, o si
hay que devolver el mejor borrador disponible anotado con sus caveats.

La clave de diseño: **separar responsabilidades** para que todo sea auditable.

| Rol | Qué hace | Hace llamadas LLM | Muta la sesión |
| --- | --- | --- | --- |
| **Actor** | Genera un `EstimationResult` (prompt conversacional v3) | Sí | No |
| **Critic** | Audita el borrador y devuelve `CriticFeedback` estructurado | Sí | No |
| **Boss** | Decide `accept` / `iterate` / `synthesize` | **No** | No |

El Actor y el Critic se pasan al Boss como **callables**, no como objetos. El
callable del actor recibe el feedback del Critic (o `None` en la primera ronda),
lo que permite re-renderizar el prompt con las correcciones entretejidas.

## El bucle

```
        ┌──────────────────────────────────────────────┐
        │                  Boss.run()                   │
        └──────────────────────────────────────────────┘
              │
   feedback   ▼            ┌───────────┐        ┌──────────┐
   (None 1ª) ──▶  actor( ) ─▶│  Actor    │───────▶│  Critic  │──▶ CriticFeedback
              ▲             └───────────┘        └──────────┘
              │                                              │
              │                                        _decide(verdict, iter_left)
              │                                              │
              │            ┌─────────────────────────────────┼──────────────┐
              │            │                                 │              │
              │        accept                            needs_iteration  reject
              │            │                                 │              │
              │            ▼                          ┌──────┴──────┐       │
              │      return (draft, trace)      iter_left>0    iter_left=0 │
              │                                 │                  │       │
              └─────────────────────────────────┘                  ▼       ▼
                          (loop con feedback)              return _synthesize_fallback
```

Pseudocódigo del orquestador (`app/services/boss.py`):

```python
for iteration in range(self.max_iterations):
    current = actor(feedback)  # borrador (ya pasó el guardrail de salida)
    review = critic(current)  # CriticFeedback estructurado
    decision = self._decide(review, iterations_left=self.max_iterations - iteration - 1)
    trace.iterations.append(ACBIteration(...))
    if decision == "accept":
        return current, trace
    if decision == "synthesize":
        return self._synthesize_fallback(current, review), trace
    feedback = review  # decision == "iterate" → re-intenta
# presupuesto agotado
return self._synthesize_fallback(current, feedback), trace
```

### Tabla de decisión (`Boss._decide`)

| `verdict` del Critic | `iterations_left > 0` | Decisión del Boss |
| --- | --- | --- |
| `accept` | — | `accept` |
| `reject` | — | `synthesize` |
| `needs_iteration` | sí | `iterate` |
| `needs_iteration` | no | `synthesize` |

## El Critic: feedback estructurado, no prosa

El Critic **nunca** devuelve texto libre. Devuelve un `CriticFeedback` validado por
Pydantic (`app/schemas/critic.py`):

- `verdict`: `accept` \| `needs_iteration` \| `reject`.
- `issues`: hasta 12 `CriticIssue`, con `category` (`math_error`, `hallucination`,
  `scope_mismatch`, `phase_imbalance`, `missing_assumption`,
  `unrealistic_estimate`, `tier_mismatch`), `severity` (`critical`/`major`/`minor`),
  `field_path` concreto (`phases[2].cost_eur`) y `suggested_fix` opcional.
- `confidence_in_review`: confianza del Critic en su propia revisión (0..100).

Dos validadores impiden verdictos incoherentes:

- `needs_iteration` exige **al menos un issue critical/major** (un minor solo no
  justifica quemar otra llamada al actor).
- `reject` exige al menos un issue que explique el rechazo.

El prompt del Critic (`app/prompts/critic/v1/system.j2`) está calibrado para evitar
falsos positivos: "ante la duda entre `accept` y `needs_iteration`, elige `accept`".

### Degradación elegante

Si el Critic falla (error de proveedor, timeout…), `Critic.review` captura la
excepción y devuelve un `CriticFeedback(verdict="accept", issues=[], confidence_in_review=0)`.
El efecto es que el Boss devuelve el borrador del actor **sin modificar**: perder la
auditoría no debe romperle el turno al usuario.

## El fallback `synthesize`

Cuando el Boss no puede aceptar (rechazo o presupuesto de iteraciones agotado),
**no descarta** el borrador. Devuelve el último resultado del actor anotado
(`Boss._synthesize_fallback`):

1. Antepone al `summary` un bloque de caveats con los issues abiertos:
   ```
   ⚠ Open caveats from independent review (loop did not fully converge):
   - [major] phase_imbalance (phases[0].cost_eur): una sola fase concentra todo el coste.
   ...
   ```
2. Reduce `confidence_pct` a `max(30, confidence_pct // 2)`.

El suelo en **30** es deliberado: por debajo, el validador de negocio exige que el
summary empiece por `"Out of scope:"`, lo que contradiría el objetivo de devolver
un borrador usable. La señal honesta de que el bucle no convergió no se esconde en
los números: viaja en `BossTrace.final_decision == "synthesize"`.

## La traza de auditoría (`ACBResponse.acb`)

La respuesta incluye `acb: BossTrace`, que el cliente puede volcar en un panel:

```json
{
  "result": { "summary": "...", "confidence_pct": 36, "phases": [ ... ] },
  "prompt_version": "v3",
  "metadata": { "project_name": "Nimbus", "assumed_team_size": 3 },
  "acb": {
    "final_decision": "synthesize",
    "iterations_run": 2,
    "iterations": [
      {
        "iteration": 0,
        "decision_after": "iterate",
        "critic_verdict": "needs_iteration",
        "critic_confidence": 60,
        "issue_summary": ["[major] phase_imbalance @ phases[0].cost_eur"]
      },
      {
        "iteration": 1,
        "decision_after": "synthesize",
        "critic_verdict": "needs_iteration",
        "critic_confidence": 55,
        "issue_summary": ["[major] phase_imbalance @ phases[0].cost_eur"]
      }
    ]
  }
}
```

## Persistencia coherente

Desde el punto de vista del usuario, un turno produce **un** mensaje del asistente.
Por eso `EstimationService.estimate_with_acb` solo añade a la sesión el resultado
final aprobado o sintetizado — los borradores intermedios se descartan. Después
aplica la compresión de memoria (anclas + resumen) y refresca `ProjectMetadata`
igual que el camino actor.

## Ejemplo de petición

```bash
SESSION=$(curl -s -X POST localhost:8000/api/v1/sessions | jq -r .session_id)

curl -s -X POST "localhost:8000/api/v1/sessions/$SESSION/estimate-acb" \
  -F 'transcript=Necesitamos un CRM para el equipo de ventas con React y Postgres.' \
  -F 'project_type=web_saas' \
  -F 'detail_level=medium' \
  -F 'output_format=phases_table' \
  -F 'tier=executive' \
  | jq '{acb_final_decision: .acb.final_decision, confidence: .result.confidence_pct}'
```

## Ficheros y configuración

| Pieza | Fichero |
| --- | --- |
| Orquestador | `app/services/boss.py` |
| Critic | `app/services/critic.py` |
| Cableado / pipeline | `app/services/estimation.py` (`estimate_with_acb`) |
| DTOs del Critic | `app/schemas/critic.py` |
| Traza | `app/schemas/acb.py` |
| Prompt del Critic | `app/prompts/critic/v1/system.j2`, `user.j2` |
| Endpoint | `app/routers/sessions.py` (`POST /{session_id}/estimate-acb`) |

Variables de entorno: `CRITIC_MODEL` (vacío = `LLM_MODEL`) y `BOSS_MAX_ITERATIONS`
(por defecto 2, debe ser ≥ 1). El `boss_max_iterations` es una cota dura: aunque el
Critic siga pidiendo iterar, el Boss sintetiza al agotar el presupuesto.

## Cómo se prueba

- `tests/test_acb_boss.py` — decisión, iteración con feedback, síntesis por rechazo
  y por presupuesto agotado, persistencia de un único turno y override de tier.
- `tests/test_critic.py` — contrato del Critic y degradación a `accept` con
  confianza 0 ante un fallo.
