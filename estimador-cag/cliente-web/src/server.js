import { createApp } from "./app.js";
import { InMemoryChunkingRepo, PgChunkingRepo } from "./chunkingRepo.js";
import { HttpConfigClient } from "./configClient.js";
import { loadConfig } from "./config.js";
import { createPool, InMemoryEstimationRepo, PgEstimationRepo } from "./db.js";
import { HttpEmbeddingsClient } from "./embeddingsClient.js";
import { HttpEstimatorClient } from "./estimatorClient.js";
import { HttpIngestionClient } from "./ingestionClient.js";
import { runMigrations } from "./migrate.js";

const config = loadConfig();

const pool = createPool(config.databaseUrl);
let repo;
let chunkingRepo;
if (pool) {
  try {
    await runMigrations(pool);
    console.log("Postgres listo y migrado.");
  } catch (error) {
    console.error("No se pudieron aplicar las migraciones:", error.message);
  }
  repo = new PgEstimationRepo(pool);
  chunkingRepo = new PgChunkingRepo(pool);
} else {
  console.warn("DATABASE_URL vacía: usando repositorios en memoria (sin persistencia).");
  repo = new InMemoryEstimationRepo();
  chunkingRepo = new InMemoryChunkingRepo();
}

const estimator = new HttpEstimatorClient({
  baseUrl: config.estimatorApiBaseUrl,
  timeoutMs: config.estimatorTimeoutMs,
});

const ingestion = new HttpIngestionClient({
  baseUrl: config.estimatorApiBaseUrl,
  timeoutMs: config.estimatorTimeoutMs,
});

const embeddings = new HttpEmbeddingsClient({
  baseUrl: config.estimatorApiBaseUrl,
  timeoutMs: config.chunkingTimeoutMs,
});

const configClient = new HttpConfigClient({
  baseUrl: config.estimatorApiBaseUrl,
  timeoutMs: config.configTimeoutMs,
  cacheTtlMs: config.configCacheTtlMs,
});

const app = createApp({
  repo,
  estimator,
  ingestion,
  chunkingRepo,
  embeddings,
  config: configClient,
});

app.listen(config.port, () => {
  console.log(
    `cliente-web escuchando en http://localhost:${config.port} (API: ${config.estimatorApiBaseUrl})`,
  );
});
