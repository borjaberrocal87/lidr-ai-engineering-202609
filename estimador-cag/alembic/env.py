"""Entorno de migraciones de Alembic para la capa de persistencia (Sesiones 6 y 8).

La URL de la base de datos se lee de ``app.config.Settings`` (no de alembic.ini)
para que el contenedor, el host de desarrollo y CI usen la misma fuente de
verdad. Las migraciones se descubren importando los módulos de modelos: cada
modelo de SQLAlchemy registrado en ``Base.metadata`` pasa a ser visible para el
autogenerate de Alembic.
"""

from __future__ import annotations

from logging.config import fileConfig

from pgvector.sqlalchemy import Vector
from sqlalchemy import engine_from_config, pool
from sqlalchemy.dialects.postgresql.base import ischema_names

import app.embedding_pipeline.store.models  # noqa: F401 — registra S8 en Base.metadata
from alembic import context
from app.config import get_settings
from app.persistence.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

# Enseña a la reflexión el tipo ``vector``. Sin esto, ``alembic check`` /
# autogenerate contra una BD que ya tiene columnas vector no sabe mapearlas de
# vuelta y produce diffs inconsistentes.
ischema_names["vector"] = Vector

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Genera SQL sin una conexión activa; lo usa ``alembic upgrade --sql``."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Aplica las migraciones contra una conexión activa a la base de datos."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
