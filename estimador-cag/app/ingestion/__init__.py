"""Subsistema de ingesta de la Sesión 6.

Tres capas conceptuales, cada una en su propio subpaquete:

* ``catalog`` — auditoría versionada de las fuentes de datos (qué ingerimos y por qué).
* ``loaders`` + ``parsers`` — bytes en crudo → list[Document] (el contrato canónico).
* ``cleaning`` + ``pii`` — validación tabular y pseudonimización GDPR.

El punto de entrada HTTP vive en ``app.routers.ingestion``; el pegamento offline
que lo une todo vive en ``app.ingestion.orchestrator``.
"""
