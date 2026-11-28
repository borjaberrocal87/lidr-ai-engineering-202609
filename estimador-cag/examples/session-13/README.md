# Session 13 — estimation graph

Session 13 re-expresses the Session 12 estimation flow as an explicit LangGraph
inside the AI service. See
[`specs/sesion-13-grafo-langgraph-estimacion.md`](../../specs/sesion-13-grafo-langgraph-estimacion.md)
for the full brief.

## Run it

```bash
# Offline mechanics smoke: in-process checkpointer + canned retrieval, no DB.
# Still needs OPENAI_API_KEY for the two LLM nodes.
uv run python scripts/run_graph_s13.py --memory --stub \
    --transcript examples/session-12/sample_transcript_simple.txt --out /tmp/graph_simple.txt

# Deliverable run: real Postgres checkpointer + real S9-S10 retrieval.
# Requires the stack up and both corpora ingested.
docker compose exec api python scripts/run_graph_s13.py \
    --transcript examples/session-12/sample_transcript_complex.txt \
    --out examples/session-13/trace_complex.txt
```

`trace_complex.txt` is the real deliverable run, reproduced in
[`docs/sesion-13-grafo-langgraph.md`](../../docs/sesion-13-grafo-langgraph.md).

## Session 13 sample transcript

`sample_transcript_marcador.txt` — a discovery meeting for **Marcador**, a live
football-scores platform (Sofascore/Livescore style) with a public **web** and a
**mobile app**. It exercises a wide, realistic decomposition: data-provider
ingestion and normalisation (with fallback), backend/API, real-time layer,
public web, mobile app, push notifications, backoffice, accounts and favourites,
monetization (ads + subscription), infrastructure/observability and product
analytics — plus peak-concurrency (500k concurrent users) and latency (<2 s)
constraints, and delivery phases.

Its real run is kept in `trace_marcador.txt` (13 components, 96 reference
matches, 474.3 h, `status="needs_review"` because one component — Match Center &
Public/Internal APIs — had no historical reference). It is a good example of the
Level 3 conditional branch: the validation flags the unbudgeted component instead
of silently inventing a number.

```bash
docker compose exec api python scripts/run_graph_s13.py \
    --transcript examples/session-13/sample_transcript_marcador.txt \
    --estimation-id s13-marcador \
    --out examples/session-13/trace_marcador.txt
```

