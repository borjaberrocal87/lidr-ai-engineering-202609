"""Servicio Critic — auditoría independiente de un `EstimationResult`.

El Critic es una función sin estado: entrada (transcript, metadata, tier,
estimación) → salida ``CriticFeedback``. NO muta la sesión y NO decide qué hacer
con su propia salida — eso es trabajo del Boss. Mantener las responsabilidades
separadas es lo que hace auditable el patrón Actor-Critic-Boss.

Un fallo dentro del Critic NO bloquea la pipeline. El Boss recibe un veredicto
sintético "accept con confianza cero" para que la estimación del actor fluya sin
modificar — degradación elegante en lugar de un error duro frente al usuario.
"""

from __future__ import annotations

import structlog

from app.prompts.loader import render_critic_prompt
from app.schemas.critic import CriticFeedback
from app.schemas.estimations import EstimationResult
from app.services.llm_wrapper import LLMWrapper
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier

log = structlog.get_logger()

CRITIC_MAX_TOKENS = 1500
CRITIC_MAX_RETRIES = 3


class Critic:
    def __init__(self, *, llm_wrapper: LLMWrapper, model: str) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model

    def review(
        self,
        *,
        transcript: str,
        metadata: ProjectMetadata,
        tier: Tier,
        result: EstimationResult,
    ) -> CriticFeedback:
        system_prompt, user_message = render_critic_prompt(
            transcript=transcript,
            metadata=metadata,
            tier=tier,
            result=result,
        )

        try:
            feedback, meta = self.llm_wrapper.complete_structured_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=CriticFeedback,
                temperature=None,
                max_tokens=CRITIC_MAX_TOKENS,
                max_retries=CRITIC_MAX_RETRIES,
                model_override=self.model,
            )
        except Exception as exc:
            log.warning(
                "critic_failed_fallback_accept",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            # Degradación: fingimos que el Critic aceptó con confianza cero para
            # que el Boss devuelva la salida del actor sin modificar.
            return CriticFeedback(verdict="accept", issues=[], confidence_in_review=0)

        log.info(
            "critic_completed",
            verdict=feedback.verdict,
            issue_count=len(feedback.issues),
            confidence=feedback.confidence_in_review,
            model=meta.get("model"),
            latency_ms=meta.get("latency_ms"),
        )
        return feedback
