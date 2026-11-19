-- Historial de expansiones incrementales del corpus (Sesión 11, directo).
-- El job real vive en la API de IA; aquí guardamos la foto de contexto que el
-- usuario envió (tipo de chunk, nº de documentos y stats ANTES del lote) para
-- que la vista /corpus muestre cómo creció el corpus.
CREATE TABLE IF NOT EXISTS corpus_index_runs (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),
  job_id           uuid NOT NULL,
  chunk_type       text NOT NULL,
  documents_total  integer NOT NULL DEFAULT 0,
  status           text NOT NULL DEFAULT 'pending',
  before_stats     jsonb,
  error            text
);

CREATE INDEX IF NOT EXISTS corpus_index_runs_created_at_idx
  ON corpus_index_runs (created_at DESC);
