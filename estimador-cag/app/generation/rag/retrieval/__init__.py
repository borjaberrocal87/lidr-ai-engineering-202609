"""Session 10 — hybrid search + cross-encoder reranking.

Two relevance techniques layered on top of the Session 9 vector retriever:

* ``fusion`` — Reciprocal Rank Fusion of the vector and lexical rankings.
* ``reranker`` — a cross-encoder that rescues precision after a wide recall.
* ``pipeline`` — the ``retrieve()`` entrypoint that composes both behind the
  ``search_mode`` (vector/hybrid) and ``rerank`` (on/off) toggles.

The package depends only on ``config``, ``foundation``/``domain`` contracts and
sibling modules under ``generation/rag`` — never on ``api`` or ``dependencies``.
"""
