import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryCorpusRunRepo } from "../src/corpusRepo.js";
import { InMemoryEstimationRepo } from "../src/db.js";

const JOB_ID = "22222222-2222-2222-2222-222222222222";

const STATS = {
  collections: [
    { collection: "budget", documents: 2, chunks: 10, hnsw_indexed: true },
    { collection: "transcript", documents: 1, chunks: 4, hnsw_indexed: false },
  ],
  total_chunks: 14,
};

const DOCUMENT = {
  budget_id: "BUDGET-2026-9001",
  client_metadata: { name: "ACME", sector: "finance", country: "ES" },
  project_summary: "Portal de pagos",
  main_technology: "python_fastapi",
  year: 2026,
  total_estimated_hours: 40,
  components: [
    {
      component_id: "PAY-001",
      name: "Reconciliation",
      description: "Conciliación nocturna",
      estimated_hours: 40,
      complexity: "medium",
    },
  ],
};

function fakeEstimator() {
  return {
    async getContext() {
      return { available_versions: ["v1"] };
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

function fakeCorpusClient({ status = "completed" } = {}) {
  const calls = { start: [], jobs: 0 };
  return {
    calls,
    async startRun({ documents, chunkType }) {
      calls.start.push({ documents, chunkType });
      return { job_id: JOB_ID, documents_total: documents.length, status: "pending" };
    },
    async getJob() {
      calls.jobs += 1;
      return {
        job_id: JOB_ID,
        status,
        documents_processed: 1,
        error_message: null,
        started_at: new Date().toISOString(),
        finished_at: status === "completed" ? new Date().toISOString() : null,
      };
    },
    async getStats() {
      return STATS;
    },
  };
}

function makeApp({ corpusIndex }) {
  return createApp({
    repo: new InMemoryEstimationRepo(),
    estimator: fakeEstimator(),
    ingestion: fakeIngestion(),
    corpusIndex,
    corpusRuns: new InMemoryCorpusRunRepo(),
  });
}

test("GET /corpus shows stats and an empty history", async () => {
  const response = await request(makeApp({ corpusIndex: fakeCorpusClient() })).get("/corpus");

  assert.equal(response.status, 200);
  assert.match(response.text, /Estado del corpus/);
  assert.match(response.text, /14 chunks/);
  assert.match(response.text, /Todavía no has añadido documentos/);
});

test("POST /corpus/runs accepts an array and redirects to the run", async () => {
  const corpusIndex = fakeCorpusClient();

  const response = await request(makeApp({ corpusIndex }))
    .post("/corpus/runs")
    .type("form")
    .send({ chunk_type: "budget_component", documents_json: JSON.stringify([DOCUMENT]) });

  assert.equal(response.status, 303);
  assert.match(response.headers.location, /^\/corpus\/runs\//);
  assert.equal(corpusIndex.calls.start.length, 1);
  assert.equal(corpusIndex.calls.start[0].documents.length, 1);
  assert.equal(corpusIndex.calls.start[0].chunkType, "budget_component");
});

test("POST /corpus/runs accepts a single object too", async () => {
  const corpusIndex = fakeCorpusClient();

  const response = await request(makeApp({ corpusIndex }))
    .post("/corpus/runs")
    .type("form")
    .send({ documents_json: JSON.stringify(DOCUMENT) });

  assert.equal(response.status, 303);
  assert.equal(corpusIndex.calls.start[0].documents.length, 1);
});

test("POST /corpus/runs rejects invalid JSON with 422", async () => {
  const response = await request(makeApp({ corpusIndex: fakeCorpusClient() }))
    .post("/corpus/runs")
    .type("form")
    .send({ documents_json: "{not json" });

  assert.equal(response.status, 422);
  assert.match(response.text, /JSON inválido/);
});

test("GET /corpus/runs/:id polls the job and shows the corpus", async () => {
  const corpusIndex = fakeCorpusClient();
  const app = makeApp({ corpusIndex });

  const created = await request(app)
    .post("/corpus/runs")
    .type("form")
    .send({ documents_json: JSON.stringify([DOCUMENT]) });
  const location = created.headers.location;

  const response = await request(app).get(location);

  assert.equal(response.status, 200);
  assert.match(response.text, /completed/);
  assert.match(response.text, /1 \/ 1/);
  assert.match(response.text, /Corpus ahora/);
  assert.ok(corpusIndex.calls.jobs >= 1);
});

test("GET /corpus/runs/:id returns 404 for an unknown run", async () => {
  const response = await request(makeApp({ corpusIndex: fakeCorpusClient() })).get(
    "/corpus/runs/33333333-3333-3333-3333-333333333333",
  );

  assert.equal(response.status, 404);
});
