export function loadConfig(env = process.env) {
  return {
    port: Number(env.PORT || 3000),
    estimatorApiBaseUrl: env.ESTIMATOR_API_BASE_URL || "http://localhost:8000",
    estimatorTimeoutMs: Number(env.ESTIMATOR_TIMEOUT_MS || 60000),
    databaseUrl: env.DATABASE_URL || "",
    nodeEnv: env.NODE_ENV || "development",
  };
}
