import { createApp } from "./app.js";
import { HttpAgentEstimationClient } from "./agentEstimationClient.js";
import { InMemoryAgentProfilesRepo, PgAgentProfilesRepo } from "./agentProfilesRepo.js";
import { InMemoryChunkingRepo, PgChunkingRepo } from "./chunkingRepo.js";
import { HttpConfigClient } from "./configClient.js";
import { loadConfig } from "./config.js";
import { HttpCorpusIndexClient } from "./corpusIndexClient.js";
import { InMemoryCorpusRunRepo, PgCorpusRunRepo } from "./corpusRepo.js";
import { createPool, InMemoryEstimationRepo, PgEstimationRepo } from "./db.js";
import { HttpEmbeddingsClient } from "./embeddingsClient.js";
import { HttpEstimatorClient } from "./estimatorClient.js";
import { HttpIngestionClient } from "./ingestionClient.js";
import { runMigrations } from "./migrate.js";
import { HttpRagEstimationClient } from "./ragEstimationClient.js";
import { InMemoryRagEstimationRepo, PgRagEstimationRepo } from "./ragEstimationRepo.js";

const config = loadConfig();

const pool = createPool(config.databaseUrl);
let repo;
let chunkingRepo;
let ragEstimations;
let corpusRuns;
let agentProfiles;
if (pool) {
  try {
    await runMigrations(pool);
    console.log("Postgres listo y migrado.");
  } catch (error) {
    console.error("No se pudieron aplicar las migraciones:", error.message);
  }
  repo = new PgEstimationRepo(pool);
  chunkingRepo = new PgChunkingRepo(pool);
  ragEstimations = new PgRagEstimationRepo(pool);
  corpusRuns = new PgCorpusRunRepo(pool);
  agentProfiles = new PgAgentProfilesRepo(pool);
} else {
  console.warn("DATABASE_URL vacía: usando repositorios en memoria (sin persistencia).");
  repo = new InMemoryEstimationRepo();
  chunkingRepo = new InMemoryChunkingRepo();
  ragEstimations = new InMemoryRagEstimationRepo();
  corpusRuns = new InMemoryCorpusRunRepo();
  agentProfiles = new InMemoryAgentProfilesRepo();
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

const ragClient = new HttpRagEstimationClient({
  baseUrl: config.estimatorApiBaseUrl,
  retrievalApiKey: config.ragRetrievalApiKey,
  estimateApiKey: config.ragEstimateApiKey,
  timeoutMs: config.ragTimeoutMs,
  generateTimeoutMs: config.ragGenerateTimeoutMs,
});

const corpusIndex = new HttpCorpusIndexClient({
  baseUrl: config.estimatorApiBaseUrl,
  timeoutMs: config.corpusIndexTimeoutMs,
});

const agentClient = new HttpAgentEstimationClient({
  baseUrl: config.estimatorApiBaseUrl,
  timeoutMs: config.agentTimeoutMs,
});

const app = createApp({
  repo,
  estimator,
  ingestion,
  chunkingRepo,
  embeddings,
  config: configClient,
  ragEstimations,
  ragClient,
  agentClient,
  agentProfiles,
  corpusIndex,
  corpusRuns,
});

app.listen(config.port, () => {
  console.log(
    `cliente-web escuchando en http://localhost:${config.port} (API: ${config.estimatorApiBaseUrl})`,
  );
});
