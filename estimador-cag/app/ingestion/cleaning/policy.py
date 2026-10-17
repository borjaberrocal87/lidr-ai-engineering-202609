"""Enrutador de la política de validación: reparar / cuarentena / descartar.

El esquema de Pandera puede lanzar en la primera comprobación que falle, pero en
producción queremos la lista *completa* de fallos: eso es lo que le dice al
operador qué familia de suciedad golpeó la corrida. Usamos ``lazy=True`` para que
Pandera recoja todos los fallos y luego enrutamos cada fila fallida a uno de tres
cubos:

* ``valid``       — pasa todas las comprobaciones.
* ``quarantined`` — fallos recuperables (p. ej. ``client_name`` NA tras la
  limpieza, metadatos opcionales ausentes). Se conserva para revisión del owner.
* ``discarded``   — fallos fatales (p. ej. ``total_amount`` negativo,
  ``budget_id`` malformado). Se registra pero no se reintenta.

El dict ``report`` es serializable a JSON para que pueda quedar junto a la fila
de ``ingestion_jobs`` en futuras migraciones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera.pandas as pa

from app.ingestion.cleaning.schemas import BudgetRecord

# Qué tipos de fallo van a cuarentena y cuáles a descarte. Cuarentena = la fila
# PUEDE volver tras revisión manual; descarte = la fila está estructuralmente mal.
QUARANTINE_CHECKS = {
    "not_nullable",  # client_name era NA
    "no_default",
    "column_in_dataframe",  # falta una columna de metadatos opcional
}
DISCARD_CHECKS = {
    "in_range",  # total_amount negativo / demasiado grande
    "less_than_or_equal_to",
    "greater_than_or_equal_to",
    "isin",  # currency fuera de {EUR,USD,GBP}
    "str_matches",  # budget_id o client_code malformado
}


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    quarantined: pd.DataFrame
    discarded: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


def validate_with_policy(df: pd.DataFrame) -> ValidationResult:
    """Ejecuta :data:`BudgetRecord` contra ``df`` y particiona las filas."""
    if df.empty:
        return ValidationResult(
            valid=df.copy(),
            quarantined=df.iloc[0:0].copy(),
            discarded=df.iloc[0:0].copy(),
            report={"input_rows": 0},
        )

    try:
        validated = BudgetRecord.validate(df, lazy=True)
        return ValidationResult(
            valid=validated,
            quarantined=df.iloc[0:0].copy(),
            discarded=df.iloc[0:0].copy(),
            report={
                "input_rows": len(df),
                "valid_rows": len(validated),
                "quarantined_rows": 0,
                "discarded_rows": 0,
                "failures_by_check": {},
            },
        )
    except pa.errors.SchemaErrors as err:
        return _route_failures(df, err)


def _route_failures(df: pd.DataFrame, err: pa.errors.SchemaErrors) -> ValidationResult:
    failure_cases = err.failure_cases.copy()
    indices_quarantine: set[int] = set()
    indices_discard: set[int] = set()
    failures_by_check: dict[str, int] = {}

    for _, row in failure_cases.iterrows():
        check_name = str(row.get("check", "unknown"))
        failures_by_check[check_name] = failures_by_check.get(check_name, 0) + 1
        idx = row.get("index")
        if idx is None or pd.isna(idx):
            continue
        try:
            row_idx = int(idx)
        except (TypeError, ValueError):
            continue

        if any(disc in check_name for disc in DISCARD_CHECKS):
            indices_discard.add(row_idx)
        elif any(quar in check_name for quar in QUARANTINE_CHECKS):
            indices_quarantine.add(row_idx)
        else:
            # Tipo de fallo desconocido: por seguridad, a cuarentena para revisión.
            indices_quarantine.add(row_idx)

    # Las filas del conjunto de descarte siempre pierden, aunque también hayan
    # disparado una comprobación de cuarentena.
    indices_quarantine -= indices_discard
    discarded_mask = df.index.isin(indices_discard)
    quarantined_mask = df.index.isin(indices_quarantine)
    valid_mask = ~(discarded_mask | quarantined_mask)

    return ValidationResult(
        valid=df.loc[valid_mask].copy(),
        quarantined=df.loc[quarantined_mask].copy(),
        discarded=df.loc[discarded_mask].copy(),
        report={
            "input_rows": len(df),
            "valid_rows": int(valid_mask.sum()),
            "quarantined_rows": int(quarantined_mask.sum()),
            "discarded_rows": int(discarded_mask.sum()),
            "failures_by_check": failures_by_check,
        },
    )
