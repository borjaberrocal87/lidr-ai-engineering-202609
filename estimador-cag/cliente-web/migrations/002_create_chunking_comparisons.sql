-- Runs persistidos del Chunking Lab (Sesión 7): permiten revisitar
-- comparativas caras (propositional/contextual_retrieval) sin re-pagarlas.
CREATE TABLE IF NOT EXISTS chunking_comparisons (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at       timestamptz NOT NULL DEFAULT now(),
  strategies       jsonb NOT NULL DEFAULT '[]'::jsonb,
  queries          jsonb NOT NULL DEFAULT '[]'::jsonb,
  top_k            integer NOT NULL DEFAULT 3,
  corpus_label     text NOT NULL DEFAULT 'budgets_sample',
  corpus_count     integer NOT NULL DEFAULT 0,
  response_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  duration_ms      integer
);

CREATE INDEX IF NOT EXISTS chunking_comparisons_created_at_idx
  ON chunking_comparisons (created_at DESC);
