-- Perfiles de agente (Sesión 12). Un preset nombrado y personalizable para el
-- agente: persona libre y knobs en un JSONB para no cambiar el esquema cuando el
-- agente gane parámetros. El avatar se guarda como data URL (PNG/JPEG/GIF/WebP)
-- para no depender de almacenamiento de ficheros en este cliente.
CREATE TABLE IF NOT EXISTS agent_profiles (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at           timestamptz NOT NULL DEFAULT now(),
  updated_at           timestamptz NOT NULL DEFAULT now(),
  name                 text NOT NULL,
  agent_type           text NOT NULL DEFAULT 'handwritten',
  persona              text,
  config               jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_default           boolean NOT NULL DEFAULT false,
  avatar_data_url      text,
  avatar_content_type  text
);

CREATE UNIQUE INDEX IF NOT EXISTS agent_profiles_name_type_idx
  ON agent_profiles (lower(name), agent_type);

CREATE INDEX IF NOT EXISTS agent_profiles_default_idx
  ON agent_profiles (is_default);
