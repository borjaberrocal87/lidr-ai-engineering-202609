"""Repositorios estrechos sobre la capa de persistencia.

Cada repositorio envuelve una única ``Session`` y expone las operaciones que su
llamador necesita de verdad — ningún tipo del ORM se filtra más allá de esta
frontera. Ambos repositorios son deliberadamente diminutos: las ejecuciones de
ingesta son poco frecuentes y las consultas del camino caliente (la búsqueda de
correspondencias) son lecturas puntuales indexadas por una clave compuesta.
"""

from app.persistence.repositories.jobs import JobsRepository
from app.persistence.repositories.mappings import MappingsRepository

__all__ = ["JobsRepository", "MappingsRepository"]
