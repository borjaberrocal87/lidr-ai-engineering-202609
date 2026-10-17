"""Reconocedores de Presidio específicos del dominio.

Presidio detecta PERSON, EMAIL, LOCATION, etc. de serie, pero el corpus del
Proyecto 2 también contiene:

* **BUDGET_ID** — ``BUDGET-YYYY-NNNN`` (año + serial de cuatro dígitos).
* **CLIENT_CODE** — ``CLI-NNNN`` (serial de cuatro dígitos).

Ambos son patrones deterministas con una forma clara — coincidencias regex de
alta precisión que deberían llegar con una puntuación de confianza cercana a
1.0. Los declaramos con ``supported_language="es"`` para que se activen junto a
la pipeline NLP en español configurada en ``analyzer.build_analyzer``.
"""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer


class BudgetIdRecognizer(PatternRecognizer):
    """Detecta identificadores de presupuesto ``BUDGET-YYYY-NNNN``."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="BUDGET_ID",
            name="BudgetIdRecognizer",
            patterns=[
                Pattern(
                    name="budget_id_pattern",
                    regex=r"\bBUDGET-\d{4}-\d{4}\b",
                    score=0.95,
                )
            ],
            supported_language="es",
        )


class ClientCodeRecognizer(PatternRecognizer):
    """Detecta códigos de cliente ``CLI-NNNN``."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="CLIENT_CODE",
            name="ClientCodeRecognizer",
            patterns=[
                Pattern(
                    name="client_code_pattern",
                    regex=r"\bCLI-\d{4}\b",
                    score=0.95,
                )
            ],
            supported_language="es",
        )
