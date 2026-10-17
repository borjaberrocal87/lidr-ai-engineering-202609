"""Esquema de Pandera v2 para el DataFrame de presupuestos limpio.

Fíjate en la ruta de importación de Pandera v2: ``import pandera.pandas as pa``.
En v1 era ``import pandera as pa``. La migración rompió muchos tutoriales; la
sesión en vivo lo señala de forma explícita.

El esquema mezcla restricciones por columna con un ``@dataframe_check`` entre
columnas que impone ``sum(phases.amount) == total_amount``. Los datos a nivel de
fase viven en un esquema paralelo ``BudgetPhases`` mantenido estrecho a
propósito; aquí solo comprobamos lo que es visible en el registro plano.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

BUDGET_ID_PATTERN = r"^BUDGET-\d{4}-\d{4}$"
CLIENT_CODE_PATTERN = r"^CLI-\d{4}$"

BudgetRecord: DataFrameSchema = DataFrameSchema(
    columns={
        "budget_id": Column(
            str,
            checks=Check.str_matches(BUDGET_ID_PATTERN),
            nullable=False,
            required=True,
        ),
        "client_name": Column(
            str,
            nullable=True,  # la limpieza mapea TBD/N/A a NA — cuarentena, no descarte
            required=True,
        ),
        "client_code": Column(
            str,
            checks=Check.str_matches(CLIENT_CODE_PATTERN),
            nullable=False,
            required=True,
        ),
        "currency": Column(
            str,
            checks=Check.isin(["EUR", "USD", "GBP"]),
            nullable=False,
            required=True,
        ),
        "total_amount": Column(
            float,
            checks=[Check.ge(0), Check.le(10_000_000)],
            nullable=False,
            required=True,
        ),
        "signed_at": Column(
            "datetime64[ns]",
            nullable=False,
            required=True,
        ),
    },
    strict=True,
    coerce=False,
)


@pa.check_input(BudgetRecord)
def _ignored() -> None:  # pragma: no cover
    """Marcador para el análisis estático de pandera. No se invoca en runtime."""
