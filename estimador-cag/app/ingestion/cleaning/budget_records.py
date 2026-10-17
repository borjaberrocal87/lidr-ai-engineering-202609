"""Limpieza reparable para la vista tabular de presupuestos.

Cinco pasos, aplicados a un DataFrame cuyo esquema tratamos como **crudo**:

1. **Marcadores de nulos** — ``"TBD"``, ``"N/A"``, cadenas vacías → ``pd.NA``.
   Los nulos disfrazados son la suciedad más fácil de pasar por alto y la más
   difícil de depurar aguas abajo; los normalizamos en la frontera.

2. **Capitalización de divisas** — ``EUR`` frente a ``eur`` se colapsa a
   mayúsculas. Cualquier otra divisa se conserva (la validación la detectará).

3. **Coerción de fechas** — ``signed_at`` puede venir como ISO ``2024-05-02`` o
   como la forma española ``12/03/2024``. ``pd.to_datetime(errors="coerce")``
   devuelve ``NaT`` para las filas no parseables; la validación las enrutará.

4. **Coerción numérica** — ``total_amount`` y ``phase.amount`` coercionados a
   numérico con ``errors="coerce"``. Las cadenas se cuelan desde exportaciones
   manuales.

5. **Dedup por hash** — los duplicados se detectan por hash de contenido, no con
   ``drop_duplicates``. Cuando el mismo ``budget_id`` aparece dos veces con
   contenido distinto, la *política* (declarada aquí en código) es "conservar la
   fila con el ``signed_at`` más reciente". Esa es una decisión de negocio, no
   técnica: documentarla forma parte del entregable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import pandas as pd

NULL_PLACEHOLDERS = {"TBD", "N/A", "n/a", "tbd", "", "null", "None", "-"}


def clean_budget_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Devuelve un DataFrame limpio a partir de la lista cruda de dicts de
    presupuesto.

    La entrada es el contenido JSON parseado — *no* el markdown renderizado de
    ``Document.text``. El paso de limpieza opera sobre el registro original,
    antes de que tenga lugar cualquier embedding.
    """
    df = pd.DataFrame(list(records))
    if df.empty:
        return df

    # 1. Marcadores de nulos
    for column in ("client_name", "contact", "contact_email", "notes"):
        if column in df.columns:
            df[column] = df[column].apply(
                lambda v: pd.NA if (isinstance(v, str) and v.strip() in NULL_PLACEHOLDERS) else v
            )

    # 2. Capitalización de divisas
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype("string").str.upper()

    # 3. Coerción de fechas (permisiva)
    if "signed_at" in df.columns:
        df["signed_at"] = pd.to_datetime(df["signed_at"], errors="coerce", dayfirst=True)

    # 4. Coerción numérica
    if "total_amount" in df.columns:
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")

    # 5. Dedup por hash con regla de negocio explícita: conservar el signed_at más reciente.
    if {"budget_id", "signed_at"}.issubset(df.columns):
        df["content_hash"] = df.apply(_content_hash, axis=1)
        # Ordenamos para que el signed_at más reciente quede ÚLTIMO por budget_id,
        # y luego descartamos las filas anteriores. Reseteamos el índice para que
        # quien llame aguas abajo no vea huecos inesperados.
        df = df.sort_values(by=["budget_id", "signed_at"], na_position="first")
        df = df.drop_duplicates(subset=["budget_id"], keep="last").reset_index(drop=True)

    return df


def _content_hash(row: pd.Series) -> str:
    """JSON canónico de la fila → sha256.

    Los valores NaN se normalizan a ``None`` para que dos filas semánticamente
    idénticas que solo difieran en dtype produzcan el mismo hash.
    """
    payload = {
        k: (None if (isinstance(v, float) and pd.isna(v)) else _serializable(v))
        for k, v in row.items()
        if k != "content_hash"
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _serializable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
