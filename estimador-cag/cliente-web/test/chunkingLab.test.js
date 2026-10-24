import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryChunkingRepo } from "../src/chunkingRepo.js";
import { InMemoryEstimationRepo } from "../src/db.js";
import { EmbeddingsError } from "../src/embeddingsClient.js";

const PAYLOAD = {
  stats_per_strategy: {
    structural: {
      strategy: "structural",
      n_chunks: 4,
      token_distribution: { min: 20, p50: 40, p95: 60, max: 70 },
      n_orphan_chunks: 0,
      n_obese_chunks: 0,
      ingestion_cost_usd: 0,
      ingestion_seconds: 0.01,
    },
  },
  queries_per_strategy: {},
};

function fakeEstimator() {
  return {
    async getContext() {
      return { available_versions: ["v1"], description_min_length: 20, description_max_length: 100 };
    },
    async estimate() {
      throw new Error("no usado");
    },
  };
}

function fakeIngestion() {
  return {
    async listSources() {
      return { version: "1", sources: [] };
    },
    async listJobs() {
      return { items: [] };
    },
  };
}

function fakeConfig() {
  return {
    async getPrimaryModel() {
      return "gpt-4o-mini";
    },
  };
}

function makeApp({ embeddings } = {}) {
  const chunkingRepo = new InMemoryChunkingRepo();
  const client = embeddings || {
    async compareChunking() {
      return PAYLOAD;
    },
  };
  const app = createApp({
    repo: new InMemoryEstimationRepo(),
    estimator: fakeEstimator(),
    ingestion: fakeIngestion(),
    chunkingRepo,
    embeddings: client,
    config: fakeConfig(),
  });
  return { app, chunkingRepo };
}

test("GET /chunking-lab shows the empty history", async () => {
  const { app } = makeApp();

  const response = await request(app).get("/chunking-lab");

  assert.equal(response.status, 200);
  assert.match(response.text, /Chunking Lab/);
  assert.match(response.text, /Aún no hay comparativas/);
});

test("GET /chunking-lab/new shows the strategy form", async () => {
  const { app } = makeApp();

  const response = await request(app).get("/chunking-lab/new");

  assert.equal(response.status, 200);
  assert.match(response.text, /Contextual retrieval/);
  assert.match(response.text, /17 presupuestos/);
});

test("POST /chunking-lab persists a run and redirects to the detail", async () => {
  const { app, chunkingRepo } = makeApp();

  const response = await request(app)
    .post("/chunking-lab")
    .type("form")
    .send({ strategies: ["structural"], queries: "OAuth\nGDPR", top_k: "2" });

  assert.equal(response.status, 303);
  assert.match(response.headers.location, /^\/chunking-lab\/[0-9a-f-]+$/);

  const runs = await chunkingRepo.list();
  assert.equal(runs.length, 1);
  assert.deepEqual(runs[0].queries, ["OAuth", "GDPR"]);
  assert.equal(runs[0].topK, 2);

  const detail = await request(app).get(response.headers.location);
  assert.equal(detail.status, 200);
  assert.match(detail.text, /Structural/);
  assert.match(detail.text, /4/);
});

test("POST /chunking-lab rejects an empty strategy selection", async () => {
  const { app } = makeApp();

  const response = await request(app)
    .post("/chunking-lab")
    .type("form")
    .send({ top_k: "3" });

  assert.equal(response.status, 422);
  assert.match(response.text, /Selecciona al menos una estrategia/);
});

test("POST /chunking-lab renders the backend error", async () => {
  const failing = {
    async compareChunking() {
      throw new EmbeddingsError("La API falló.", { status: 500 });
    },
  };
  const { app } = makeApp({ embeddings: failing });

  const response = await request(app)
    .post("/chunking-lab")
    .type("form")
    .send({ strategies: ["contextual_retrieval"], top_k: "3" });

  assert.equal(response.status, 502);
  assert.match(response.text, /La API falló/);
});

test("GET /chunking-lab/:id returns 404 for an unknown run", async () => {
  const { app } = makeApp();

  const response = await request(app).get("/chunking-lab/desconocido");

  assert.equal(response.status, 404);
});
