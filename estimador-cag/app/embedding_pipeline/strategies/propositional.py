"""Propositional chunker.

An LLM decomposes each component into self-contained atomic propositions; each
proposition becomes its own chunk. Precise and great for fact-level retrieval,
but expensive: one LLM call per component. We capture token usage and report the
extra cost.

Requires an OpenAI key. Uses JSON mode + Pydantic validation (no Instructor,
which is incompatible with our OpenAI SDK version).
"""

from __future__ import annotations

import time

import structlog
from openai import OpenAI
from pydantic import BaseModel, Field

from app.embedding_pipeline.base import Chunker, count_tokens, emit_chunking_done
from app.embedding_pipeline.schemas import Budget, Chunk
from app.embedding_pipeline.structural import component_metadata, render_component_text

log = structlog.get_logger()

# gpt-4o-mini pricing (input/output). CHANGES OVER TIME.
GPT4O_MINI_INPUT_PER_MILLION_USD = 0.15
GPT4O_MINI_OUTPUT_PER_MILLION_USD = 0.60

_SYSTEM_PROMPT = (
    "Decompose the following content into clear, simple, self-contained "
    "propositions. Each proposition must:\n"
    "1. Express a single atomic fact.\n"
    "2. Be understandable on its own, without the surrounding text.\n"
    "3. Resolve pronouns and references to the explicit entity they refer to "
    "(e.g. replace 'it' with the component name).\n"
    "Return the list of propositions."
)


class _Propositions(BaseModel):
    propositions: list[str] = Field(default_factory=list)


class PropositionalChunker(Chunker):
    strategy_name = "propositional"

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        calls = 0
        cost = 0.0
        for budget in budgets:
            for component in budget.components:
                text = render_component_text(budget, component)
                props, usage = self._decompose(text)
                calls += 1
                cost += usage
                if not props:
                    # Fallback: keep the component as a single chunk so a failed
                    # decomposition never drops content silently.
                    props = [text]
                for j, prop in enumerate(props):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{component.component_id}::prop{j}",
                            text=prop,
                            metadata={**component_metadata(budget, component), "proposition": j},
                            token_count=count_tokens(prop),
                        )
                    )
        self.last_extra_api_calls = calls
        self.last_extra_cost_usd = cost
        emit_chunking_done(
            strategy=self.strategy_name,
            chunks=chunks,
            n_input_documents=len(budgets),
            extra_api_calls=calls,
            extra_cost_usd=cost,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return chunks

    def _decompose(self, text: str) -> tuple[list[str], float]:
        """Return (propositions, cost_usd). Logs and degrades on any API error."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{_SYSTEM_PROMPT}\n"
                            'Devuelve un objeto JSON con la forma {"propositions": ["..."]}.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            )
            content = completion.choices[0].message.content or "{}"
            result = _Propositions.model_validate_json(content)
        except Exception as exc:
            log.warning(
                "propositional_call_failed", error_type=type(exc).__name__, error=str(exc)[:200]
            )
            return [], 0.0
        usage = completion.usage
        if usage is None:
            return result.propositions, 0.0
        cost = (
            usage.prompt_tokens / 1_000_000 * GPT4O_MINI_INPUT_PER_MILLION_USD
            + usage.completion_tokens / 1_000_000 * GPT4O_MINI_OUTPUT_PER_MILLION_USD
        )
        return result.propositions, cost
