import { createApp } from "./app.js";
import { loadConfig } from "./config.js";
import { createPool, InMemoryEstimationRepo, PgEstimationRepo } from "./db.js";
import { HttpEstimatorClient } from "./estimatorClient.js";
import { runMigrations } from "./migrate.js";

const config = loadConfig();

const pool = createPool(config.databaseUrl);
let repo;
if (pool) {
  try {
    await runMigrations(pool);
    console.log("Postgres listo y migrado.");
  } catch (error) {
    console.error("No se pudieron aplicar las migraciones:", error.message);
  }
  repo = new PgEstimationRepo(pool);
} else {
  console.warn("DATABASE_URL vacía: usando repositorio en memoria (sin persistencia).");
  repo = new InMemoryEstimationRepo();
}

const estimator = new HttpEstimatorClient({
  baseUrl: config.estimatorApiBaseUrl,
  timeoutMs: config.estimatorTimeoutMs,
});

const app = createApp({ repo, estimator });

app.listen(config.port, () => {
  console.log(
    `cliente-web escuchando en http://localhost:${config.port} (API: ${config.estimatorApiBaseUrl})`,
  );
});
