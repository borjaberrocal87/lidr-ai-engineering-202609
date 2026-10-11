"""Detección de anclas — qué turnos deben sobrevivir al desalojo.

Un turno se convierte en ancla cuando porta un compromiso durable que la
conversación no puede permitirse olvidar: acuerdos firmados, alcance congelado,
contexto regulatorio o legal, presupuestos bloqueados. El detector es
deliberadamente conservador — los falsos negativos solo degradan el turno a la
ventana deslizante normal (que luego se resume), mientras que los falsos
positivos inflan el prompt porque las anclas nunca se desalojan.

Conviven dos estrategias:

- ``"heuristic"`` (por defecto): regex barata sobre una lista curada de frases.
  Sin llamada al LLM, determinista, fácil de razonar en directo.
- ``"llm"``: un clasificador binario vía Instructor (ver ``classify_anchor``).
  Robusto a paráfrasis pero añade una llamada LLM barata por turno — opt-in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.sessions.models import Message

log = structlog.get_logger()

# Biblioteca curada de frases. Cada entrada es una regex compilada que captura
# una frase verbal distintiva, un término legal/compliance o un compromiso
# explícito. La lista es intencionadamente corta — preferimos perder un ancla
# antes que anclar de más.
_HEURISTIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nda", re.compile(r"\b(nda|non[- ]?disclosure|under embargo|legal hold)\b", re.IGNORECASE)),
    (
        "signed_contract",
        re.compile(
            r"\b(signed|countersigned)\s+(the\s+)?(contract|sow|msa|agreement)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "scope_frozen",
        re.compile(r"\b(scope|backlog)\s+(is\s+)?(frozen|locked|final|fixed)\b", re.IGNORECASE),
    ),
    (
        "budget_locked",
        re.compile(
            r"\b(budget|cap|ceiling)\s+(is\s+)?(locked|fixed|approved|capped)\s+at\b",
            re.IGNORECASE,
        ),
    ),
    (
        "compliance",
        re.compile(r"\b(hipaa|gdpr|sox|pci[- ]?dss|fda|iso[- ]?27001)\b", re.IGNORECASE),
    ),
    ("deadline_hard", re.compile(r"\bhard\s+deadline\b|\bmust\s+go\s+live\s+by\b", re.IGNORECASE)),
    ("contractual", re.compile(r"\bcontractually\s+(bound|required|obliged)\b", re.IGNORECASE)),
    (
        "explicit_commitment",
        re.compile(r"\b(we|the (client|customer))\s+(agreed|committed)\s+to\b", re.IGNORECASE),
    ),
)


@dataclass
class AnchorMatch:
    """Resultado de una comprobación de ancla. Porta los nombres de regla que
    dispararon para que el log estructurado pueda atribuir la decisión."""

    is_anchor: bool
    matched_rules: list[str] = field(default_factory=list)


class _AnchorClassification(BaseModel):
    """Esquema Pydantic usado por el detector basado en LLM."""

    is_anchor: bool = Field(
        description=(
            "True if this user turn introduces a durable commitment, legal/"
            "compliance constraint, frozen scope, or locked budget that the "
            "conversation must not forget across turns."
        )
    )
    reason: str = Field(default="", max_length=200)


class AnchorDetector:
    """Decide si un turno merece sobrevivir al desalojo.

    El detector inspecciona el turno del *usuario* — la respuesta del asistente
    es función suya, así que basta con marcar una vez por par.
    """

    def __init__(
        self,
        *,
        mode: Literal["heuristic", "llm"] = "heuristic",
        llm_wrapper: object | None = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.mode = mode
        self.llm_wrapper = llm_wrapper
        self.llm_model = llm_model

    def detect(self, message: Message) -> AnchorMatch:
        """Devuelve la decisión de ancla más las reglas que la dispararon."""
        if self.mode == "llm":
            return self._detect_llm(message)
        return self._detect_heuristic(message)

    # -- implementaciones -------------------------------------------------

    @staticmethod
    def _detect_heuristic(message: Message) -> AnchorMatch:
        content = message.content or ""
        matched = [name for name, pattern in _HEURISTIC_PATTERNS if pattern.search(content)]
        if matched:
            log.info(
                "anchor_detected",
                strategy="heuristic",
                rules=matched,
                content_chars=len(content),
            )
            return AnchorMatch(is_anchor=True, matched_rules=matched)
        return AnchorMatch(is_anchor=False)

    def _detect_llm(self, message: Message) -> AnchorMatch:
        if self.llm_wrapper is None:
            # Fallback: comportarse como heurística si no se cableó wrapper.
            return self._detect_heuristic(message)
        system_prompt = (
            "You are a classifier. Decide whether the user turn introduces a "
            "durable commitment about the project — signed agreements, frozen "
            "scope, locked budget, or legal/compliance context (NDA, HIPAA, "
            "GDPR, etc). Only return is_anchor=true if the turn would harm the "
            "conversation if forgotten. Otherwise return false. Reason is one short sentence."
        )
        user_message = f"User turn:\n{message.content}"
        try:
            classification, _meta = self.llm_wrapper.complete_structured_messages(  # type: ignore[attr-defined]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=_AnchorClassification,
                temperature=None,
                max_tokens=200,
                max_retries=1,
                model_override=self.llm_model,
            )
        except Exception as exc:
            log.warning(
                "anchor_llm_failed_fallback_heuristic",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return self._detect_heuristic(message)
        if classification.is_anchor:
            log.info("anchor_detected", strategy="llm", reason=classification.reason[:120])
            return AnchorMatch(is_anchor=True, matched_rules=[f"llm:{classification.reason[:60]}"])
        return AnchorMatch(is_anchor=False)
