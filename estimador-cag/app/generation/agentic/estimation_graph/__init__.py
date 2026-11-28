"""Estimation flow re-expressed as an explicit LangGraph (Session 13).

The graph lives beside the Session 12 agent (``estimation_agent``) and keeps the
same layering contract: it must not import ``app.dependencies`` or
``app.generation.rag``. Every node receives its dependencies (LLM wrapper and
budget-search backend) injected from the composition root.
"""

from __future__ import annotations
