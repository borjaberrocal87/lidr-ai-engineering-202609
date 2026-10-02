-- Estimaciones persistidas por el cliente de negocio.
CREATE TABLE IF NOT EXISTS estimations (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at    timestamptz NOT NULL DEFAULT now(),
  project_type  text NOT NULL,
  detail_level  text NOT NULL,
  output_format text NOT NULL,
  prompt_version text NOT NULL,
  cached        boolean NOT NULL DEFAULT false,
  description   text NOT NULL,
  result_json   jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS estimations_created_at_idx ON estimations (created_at DESC);
