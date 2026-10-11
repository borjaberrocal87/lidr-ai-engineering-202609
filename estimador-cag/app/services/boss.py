"""Orquestador Boss — coordina Actor y Critic.

El Boss es una máquina de estados minúscula:

    actor_call → critic_review → decide
                                  ├── accept      → return
                                  ├── needs_iteration AND iterations_left
                                  │                  → re-call actor with feedback
                                  │                  → loop
                                  └── (out of iterations or reject)
                                                   → synthesize fallback

El Boss NO hace llamadas LLM propias — solo elige qué hacer a continuación. Eso
mantiene la lógica de decisión demostrable: cada elección es trazable en
``BossTrace.iterations`` y reproducible a partir de las entradas.

El Actor se pasa como callable en lugar de objeto. La firma es::

    actor(critic_feedback: CriticFeedback | None) -> EstimationResult

para que el Boss pueda re-invocarlo en la iteración N+1 con el feedback de la N.
El cableado vive en ``EstimationService.estimate_with_acb``.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from app.schemas.acb import ACBIteration, BossDecision, BossTrace
from app.schemas.critic import CriticFeedback
from app.schemas.estimations import EstimationResult

log = structlog.get_logger()

ActorCallable = Callable[[CriticFeedback | None], EstimationResult]
CriticCallable = Callable[[EstimationResult], CriticFeedback]


class Boss:
    """Orquestador sin estado. Un ``run`` por estimación."""

    def __init__(self, *, max_iterations: int = 2) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations

    def run(
        self,
        *,
        actor: ActorCallable,
        critic: CriticCallable,
    ) -> tuple[EstimationResult, BossTrace]:
        trace = BossTrace(final_decision="accept", iterations_run=0)
        feedback: CriticFeedback | None = None
        current: EstimationResult | None = None

        for iteration in range(self.max_iterations):
            current = actor(feedback)
            review = critic(current)

            decision = self._decide(review, iterations_left=(self.max_iterations - iteration - 1))
            trace.iterations.append(
                ACBIteration(
                    iteration=iteration,
                    decision_after=decision,
                    critic_verdict=review.verdict,
                    critic_confidence=review.confidence_in_review,
                    issue_summary=[
                        f"[{i.severity}] {i.category} @ {i.field_path}" for i in review.issues[:5]
                    ],
                )
            )
            trace.iterations_run = iteration + 1
            log.info(
                "boss_iteration",
                iteration=iteration,
                decision=decision,
                verdict=review.verdict,
                issues=len(review.issues),
            )

            if decision == "accept":
                trace.final_decision = "accept"
                return current, trace
            if decision == "synthesize":
                trace.final_decision = "synthesize"
                return self._synthesize_fallback(current, review), trace

            # decision == "iterate" → repetimos con feedback
            feedback = review

        # Bucle agotado: último resultado del actor con el Critic aún señalando.
        trace.final_decision = "synthesize"
        log.info("boss_iteration_budget_exhausted", iterations=self.max_iterations)
        if current is None:  # pragma: no cover — max_iterations >= 1 lo garantiza
            raise RuntimeError("Boss.run no ejecutó ninguna iteración")
        return self._synthesize_fallback(current, feedback), trace

    # -- helpers de decisión ----------------------------------------------

    @staticmethod
    def _decide(review: CriticFeedback, *, iterations_left: int) -> BossDecision:
        if review.verdict == "accept":
            return "accept"
        if review.verdict == "reject":
            return "synthesize"
        # needs_iteration
        if iterations_left > 0:
            return "iterate"
        return "synthesize"

    @staticmethod
    def _synthesize_fallback(
        last_result: EstimationResult,
        last_feedback: CriticFeedback | None,
    ) -> EstimationResult:
        """Cuando el Boss no puede aceptar, devuelve el último borrador del actor
        anotado con los issues abiertos del Critic y una confianza reducida —
        nunca un sobre vacío.

        Razón: descartar el borrador esconde la mejor respuesta disponible tras
        un placeholder ("Out of scope"). La señal honesta de que el bucle no
        convergió ya viaja en ``BossTrace.final_decision == "synthesize"`` y en
        el panel de auditoría; el usuario recibe más valor con una estimación con
        caveats que con números a cero.
        """
        if last_feedback is None or not last_feedback.issues:
            # Rama defensiva: no hay feedback que adjuntar. Devolvemos el
            # borrador con una nota mínima para que se vea que salió del camino
            # synthesize.
            note = "⚠ Open caveats from independent review: (no detail available)\n\n"
            new_summary = (note + last_result.summary)[:1200]
            return last_result.model_copy(update={"summary": new_summary})

        issue_lines = [
            f"- [{issue.severity}] {issue.category} ({issue.field_path}): {issue.description}"
            for issue in last_feedback.issues
        ]
        caveats_block = (
            "⚠ Open caveats from independent review (loop did not fully converge):\n"
            + "\n".join(issue_lines)
            + "\n\n"
        )
        new_summary = (caveats_block + last_result.summary)[:1200]
        # Suelo en 30 para seguir por encima de ``LOW_CONFIDENCE_THRESHOLD``: bajar
        # dispararía el validador que exige el prefijo "Out of scope:", lo que
        # contradice el objetivo de devolver un borrador usable y anotado.
        reduced_confidence = max(30, last_result.confidence_pct // 2)
        return last_result.model_copy(
            update={
                "summary": new_summary,
                "confidence_pct": reduced_confidence,
            }
        )
