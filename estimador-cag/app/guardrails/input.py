"""Guardrails de entrada: tres capas antes de llamar al LLM.

1. Moderación (API de OpenAI): bloquea odio, violencia, contenido sexual, etc.
2. Heurísticas de prompt-injection (regex): segunda línea barata contra
   "ignora las instrucciones anteriores".
3. Heurísticas de PII (regex): email, IBAN y teléfono. No pretende ser un
   redactor exhaustivo, sino demostrar el patrón.

Política: cualquiera de las tres lanza ``InputGuardrailViolation`` (excepción,
nunca fix-and-retry). El ``reason`` permite a la capa HTTP elegir el código y a
los clientes renderizar un mensaje accionable.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

Reason = Literal["moderation", "prompt_injection", "pii"]


class InputGuardrailViolation(Exception):
    """Se lanza cuando una capa de entrada rechaza la descripción."""

    def __init__(self, message: str, *, reason: Reason) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason


_PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"ignore\s+(previous|prior|all|the)\s+(instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(r"ignora\s+(las\s+)?(instrucciones|reglas|indicaciones)\s+anterior", re.IGNORECASE),
    re.compile(r"olvida\s+(todo|todas|lo\s+anterior)", re.IGNORECASE),
    re.compile(r"</?\s*(system|instructions?|prompt)\s*>", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*[:.\-]", re.IGNORECASE),
    re.compile(r"nuevas?\s+instrucciones?\s*[:.\-]", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bahora\s+eres\b", re.IGNORECASE),
    re.compile(
        r"\bdisregard\b.{0,40}\b(instructions?|prompts?|rules?|context|previous|prior)",
        re.IGNORECASE | re.DOTALL,
    ),
]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
# Teléfono: internacional +XX con 9-12 dígitos, o nacional de 9+ dígitos.
# Conservador a propósito para no marcar fechas ni números de versión.
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,12}\d")


def check_input(description: str, *, openai_client: Any | None = None) -> None:
    """Ejecuta las tres capas en orden y lanza en la primera violación.

    ``openai_client`` es opcional para que los tests y los entornos sin OpenAI
    se salten la moderación y solo ejerciten las capas regex.
    """
    if openai_client is not None:
        _check_moderation(description, openai_client)
    _check_prompt_injection(description)
    _check_pii(description)


def _check_moderation(description: str, openai_client: Any) -> None:
    try:
        response = openai_client.moderations.create(input=description)
    except Exception as exc:  # fallo de red/auth: se registra y se sigue (fail-open)
        logger.warning(
            "moderation_call_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return

    result = response.results[0]
    if getattr(result, "flagged", False):
        categories = _extract_flagged_categories(result)
        logger.info("moderation_flagged", categories=categories)
        raise InputGuardrailViolation(
            f"Entrada marcada por moderación: {', '.join(categories) or 'sin detalle'}",
            reason="moderation",
        )


def _extract_flagged_categories(result: Any) -> list[str]:
    categories = getattr(result, "categories", None)
    if categories is None:
        return []
    data = categories.model_dump() if hasattr(categories, "model_dump") else categories.__dict__
    return [name for name, flagged in data.items() if flagged]


def _check_prompt_injection(description: str) -> None:
    for pattern in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(description)
        if match:
            logger.info(
                "prompt_injection_detected",
                pattern=pattern.pattern,
                match=match.group(0)[:80],
            )
            raise InputGuardrailViolation(
                f"Se detectó texto sospechoso tipo instrucción: {match.group(0)[:80]!r}",
                reason="prompt_injection",
            )


def _check_pii(description: str) -> None:
    if _EMAIL_RE.search(description):
        raise InputGuardrailViolation(
            "Se detectó un email en la descripción: elimina los datos personales.",
            reason="pii",
        )
    if _IBAN_RE.search(description):
        raise InputGuardrailViolation(
            "Se detectó un IBAN en la descripción: elimina los datos personales.",
            reason="pii",
        )
    if _PHONE_RE.search(description):
        raise InputGuardrailViolation(
            "Se detectó un teléfono en la descripción: elimina los datos personales.",
            reason="pii",
        )
