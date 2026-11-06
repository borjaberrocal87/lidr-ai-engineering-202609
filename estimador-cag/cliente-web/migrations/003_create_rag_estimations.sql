-- Runs del wizard RAG (Sesión 9). Cada fila persiste la transcripción y el
-- resultado de cada etapa para poder revisar, re-ejecutar y verificar sin
-- volver a pagar las etapas anteriores.
CREATE TABLE IF NOT EXISTS rag_estimations (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),
  status           text NOT NULL DEFAULT 'draft',
  current_step     text NOT NULL DEFAULT 'brief',
  transcript       text NOT NULL,
  idempotency_key  text,
  brief            jsonb,
  retrieval_text   text,
  retrieval        jsonb,
  assembly         jsonb,
  generation       jsonb,
  verification     jsonb,
  error            text
);

CREATE INDEX IF NOT EXISTS rag_estimations_created_at_idx
  ON rag_estimations (created_at DESC);

CREATE INDEX IF NOT EXISTS rag_estimations_status_idx
  ON rag_estimations (status);
