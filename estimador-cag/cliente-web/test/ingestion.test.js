import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryEstimationRepo } from "../src/db.js";
import { IngestionError } from "../src/ingestionClient.js";

const JOB_ID = "11111111-1111-1111-1111-111111111111";

const CATALOG = {
  version: "1.1.0",
  sources: [
    {
      name: "presupuestos_json",
      description: "Presupuestos firmados en JSON.",
      format: "json",
      decision: "include",
      decision_reason: "",
      has_pii: true,
    },
    {
      name: "rate_card_xlsx",
      description: "Tarifas internas.",
      format: "xlsx",
      decision: "exclude",
      decision_reason: "obsoleta",
      has_pii: false,
    },
  ],
};

function fakeEstimator() {
  return {
    async getContext() {
      return { available_versions: ["v1"] };
    },
    async estimate() {
      throw new Error("no usado en estos tests");
    },
  };
}

function fakeIngestion({ startError } = {}) {
  const jobs = [];
  return {
    async listSources() {
      return CATALOG;
    },
    async listJobs() {
      return { items: jobs, limit: 10, offset: 0 };
    },
    async startRun(sourceName) {
      if (startError) throw startError;
      const job = {
        job_id: JOB_ID,
        source_name: sourceName,
        status: "pending",
        documents_count: 0,
        error_message: null,
        started_at: new Date().toISOString(),
        finished_at: null,
      };
      jobs.push(job);
      return job;
    },
    async getJob(jobId) {
      const job = jobs.find((item) => item.job_id === jobId);
      if (!job) {
        throw new IngestionError("El job no existe.", { status: 404, reason: "not_found" });
      }
      return job;
    },
  };
}

function makeApp(ingestion) {
  return createApp({
    repo: new InMemoryEstimationRepo(),
    estimator: fakeEstimator(),
    ingestion,
  });
}

test("GET /ingestion lists catalog sources", async () => {
  const response = await request(makeApp(fakeIngestion())).get("/ingestion");

  assert.equal(response.status, 200);
  assert.match(response.text, /Ingesta de datos/);
  assert.match(response.text, /presupuestos_json/);
  assert.match(response.text, /rate_card_xlsx/);
  assert.match(response.text, /action="\/ingestion\/runs"/);
});

test("POST /ingestion/runs starts a run and redirects to the job", async () => {
  const response = await request(makeApp(fakeIngestion()))
    .post("/ingestion/runs")
    .type("form")
    .send({ source_name: "presupuestos_json" });

  assert.equal(response.status, 303);
  assert.match(response.headers.location, /\/ingestion\/jobs\//);
});

test("POST /ingestion/runs surfaces source_not_included as 400", async () => {
  const ingestion = fakeIngestion({
    startError: new IngestionError("La fuente no está incluida.", {
      status: 400,
      reason: "source_not_included",
    }),
  });

  const response = await request(makeApp(ingestion))
    .post("/ingestion/runs")
    .type("form")
    .send({ source_name: "rate_card_xlsx" });

  assert.equal(response.status, 400);
  assert.match(response.text, /no está incluida/);
});

test("GET /ingestion/jobs/:id renders the job", async () => {
  const ingestion = fakeIngestion();
  const app = makeApp(ingestion);
  await ingestion.startRun("presupuestos_json");

  const response = await request(app).get(`/ingestion/jobs/${JOB_ID}`);

  assert.equal(response.status, 200);
  assert.match(response.text, /Job de ingesta/);
  assert.match(response.text, /presupuestos_json/);
});

test("GET /ingestion/jobs/:id returns 404 for an unknown job", async () => {
  const response = await request(makeApp(fakeIngestion())).get(
    "/ingestion/jobs/00000000-0000-0000-0000-000000000000",
  );

  assert.equal(response.status, 404);
});
