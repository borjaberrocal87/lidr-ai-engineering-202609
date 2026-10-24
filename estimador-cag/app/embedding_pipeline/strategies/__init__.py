"""Chunking strategies for the Session 7 comparison.

Seven strategies behind the common :class:`~app.embedding_pipeline.base.Chunker`
interface (the structural chunker lives in ``app.embedding_pipeline.structural``).
Re-exported here so ``from app.embedding_pipeline.strategies import *`` is a
one-line pre-flight check that every strategy imports cleanly.
"""

from app.embedding_pipeline.strategies.contextual_retrieval import ContextualRetrievalChunker
from app.embedding_pipeline.strategies.fixed_size import FixedSizeChunker
from app.embedding_pipeline.strategies.hierarchical import HierarchicalChunker
from app.embedding_pipeline.strategies.propositional import PropositionalChunker
from app.embedding_pipeline.strategies.recursive import RecursiveChunker
from app.embedding_pipeline.strategies.semantic import SemanticChunker
from app.embedding_pipeline.strategies.sentence_window import SentenceWindowChunker

__all__ = [
    "ContextualRetrievalChunker",
    "FixedSizeChunker",
    "HierarchicalChunker",
    "PropositionalChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceWindowChunker",
]
