"""Store vectorial mínimo sobre PostgreSQL + pgvector (Sesión 08).

Persistencia de ``documents`` y ``chunks``, sin índice vectorial, sin filtros de
metadata y sin búsqueda híbrida: el sequential scan es el baseline del directo.
"""
