export function loadConfig(env = process.env) {
  return {
    port: Number(env.PORT || 3000),
    estimatorApiBaseUrl: env.ESTIMATOR_API_BASE_URL || "http://localhost:8000",
    estimatorTimeoutMs: Number(env.ESTIMATOR_TIMEOUT_MS || 60000),
    databaseUrl: env.DATABASE_URL || "",
    chunkingTimeoutMs: Number(env.CHUNKING_TIMEOUT_MS || 600000),
    configTimeoutMs: Number(env.CONFIG_TIMEOUT_MS || 3000),
    configCacheTtlMs: Number(env.CONFIG_CACHE_TTL_MS || 15000),
    nodeEnv: env.NODE_ENV || "development",
  };
}
