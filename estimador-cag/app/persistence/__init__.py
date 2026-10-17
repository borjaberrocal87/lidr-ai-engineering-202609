"""Capa de persistencia del subsistema de ingesta de la Sesión 6.

Contiene el motor de SQLAlchemy, la ``Base`` declarativa y los modelos de fila
que respaldan la tabla de correspondencias de pseudonimización y el registro de
jobs de ingesta. Las capas superiores consumen repositorios estrechos de
``app.persistence.repositories``; nunca ven tipos de SQLAlchemy directamente.
"""

from app.persistence.database import (
    SessionLocal,
    create_engine_from_settings,
    get_session,
)
from app.persistence.models import Base, IngestionJobRow, PseudonymMappingRow

__all__ = [
    "Base",
    "IngestionJobRow",
    "PseudonymMappingRow",
    "SessionLocal",
    "create_engine_from_settings",
    "get_session",
]
