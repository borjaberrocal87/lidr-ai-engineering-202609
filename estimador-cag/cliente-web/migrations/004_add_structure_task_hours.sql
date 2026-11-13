-- Sesión 10: el flujo separa estructura y horas.
-- `generation` guarda la respuesta del LLM (estructura sin horas);
-- `structure` guarda la estructura revisada por la persona;
-- `task_hours` guarda el consenso vectorial por tarea (horas + fiabilidad + vecinos);
-- `verification` sigue siendo el desglose final confirmado (horas + tarifa + coste).
ALTER TABLE rag_estimations
  ADD COLUMN IF NOT EXISTS structure jsonb;

ALTER TABLE rag_estimations
  ADD COLUMN IF NOT EXISTS task_hours jsonb;
