"""Capa de limpieza y validación.

Dos pasadas:

* ``budget_records.clean_budget_records`` — correcciones a nivel de pandas
  (marcadores de nulos, capitalización de divisas, coerción permisiva, dedup por
  hash).
* ``policy.validate_with_policy`` — comprobaciones de campo y entre columnas a
  nivel de Pandera, enrutando los fallos a las particiones valid / quarantined /
  discarded.

La separación es deliberada: la limpieza da forma a los datos, la validación
los controla.
"""

from app.ingestion.cleaning.budget_records import clean_budget_records
from app.ingestion.cleaning.policy import ValidationResult, validate_with_policy
from app.ingestion.cleaning.schemas import BudgetRecord

__all__ = [
    "BudgetRecord",
    "ValidationResult",
    "clean_budget_records",
    "validate_with_policy",
]
