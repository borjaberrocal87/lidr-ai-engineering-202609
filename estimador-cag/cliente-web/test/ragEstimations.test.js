import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryEstimationRepo } from "../src/db.js";
import { InMemoryRagEstimationRepo } from "../src/ragEstimationRepo.js";
import { RagAuthError } from "../src/ragEstimationClient.js";

const TRANSCRIPT = "x".repeat(120);

const BRIEF = {
  functionality: "B2B payments platform",
  technologies: ["python", "stripe"],
  sector: "finance",
  scale: "medium",
  country: "ES",
  regulations: ["PSD2"],
  constraints: ["6 months"],
};

const CHUNK = {
  id: 1,
  document_id: 2,
  content: "[Module: Auth] OAuth login",
  sector: "finance",
  project_year: 2024,
  chunk_type: "historical_task",
  distance: 0.2,
};

const RETRIEVAL = {
  chunks: [CHUNK],
  low_confidence: false,
  candidates_evaluated: 5,
};

const ASSEMBLY = {
  context_block: '<source id="1" sector="finance">OAuth</source>',
  kept_chunks: [CHUNK],
  dropped_count: 0,
  token_count: 12,
};

const ESTIMATE = {
  total_engineer_days: 10,
  modules: [
    {
      name: "Auth",
      description: "Login",
      tasks: [{ name: "OAuth", description: null, engineer_days: 10, sources: [1] }],
    },
  ],
  duration_weeks: 2,
  sources: [{ source_id: 1, relevance: "primary", used_for: "auth" }],
  assumptions: [],
  confidence: "high",
  reasoning: "histórico de auth",
  insufficient_context_explanation: null,
};

const GENERATION = {
  estimate: ESTIMATE,
  fabricated_source_ids: [],
  coherent: true,
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
  return { async getPrimaryModel() { return "gpt-4o-mini"; } };
}

function fakeRagClient(overrides = {}) {
  return {
    async brief() {
      return { brief: BRIEF, retrieval_text: "B2B payments platform with python" };
    },
    async retrieve() {
      return RETRIEVAL;
    },
    async assemble() {
      return ASSEMBLY;
    },
    async generate() {
      return GENERATION;
    },
    ...overrides,
  };
}

function makeApp({ ragClient } = {}) {
  const ragEstimations = new InMemoryRagEstimationRepo();
  const app = createApp({
    repo: new InMemoryEstimationRepo(),
    estimator: fakeEstimator(),
    ingestion: fakeIngestion(),
    chunkingRepo: null,
    embeddings: null,
    config: fakeConfig(),
    ragEstimations,
    ragClient: ragClient || fakeRagClient(),
  });
  return { app, ragEstimations };
}

async function seedRun(ragEstimations, patch) {
  const id = await ragEstimations.create({ transcript: TRANSCRIPT });
  await ragEstimations.update(id, patch);
  return id;
}

test("GET /rag-estimations shows the empty history", async () => {
  const { app } = makeApp();

  const response = await request(app).get("/rag-estimations");

  assert.equal(response.status, 200);
  assert.match(response.text, /RAG Estimations/);
  assert.match(response.text, /Aún no hay runs/);
});

test("GET /rag-estimations/new shows the transcript form", async () => {
  const { app } = makeApp();

  const response = await request(app).get("/rag-estimations/new");

  assert.equal(response.status, 200);
  assert.match(response.text, /Nueva estimación RAG/);
  assert.match(response.text, /Mínimo 100 caracteres/);
});

test("POST /rag-estimations rejects a short transcript", async () => {
  const { app } = makeApp();

  const response = await request(app)
    .post("/rag-estimations")
    .type("form")
    .send({ transcript: "corta" });

  assert.equal(response.status, 422);
  assert.match(response.text, /al menos 100 caracteres/);
});

test("POST /rag-estimations creates a run and extracts the brief", async () => {
  const { app, ragEstimations } = makeApp();

  const response = await request(app)
    .post("/rag-estimations")
    .type("form")
    .send({ transcript: TRANSCRIPT });

  assert.equal(response.status, 303);
  const run = (await ragEstimations.list())[0];
  assert.equal(run.brief.functionality, BRIEF.functionality);
  assert.match(response.headers.location, new RegExp(`/rag-estimations/${run.id}\\?step=brief`));
});

test("GET show renders the brief and retrieval text", async () => {
  const { app, ragEstimations } = makeApp();
  const id = await seedRun(ragEstimations, {
    brief: BRIEF,
    retrievalText: "B2B payments platform",
    status: "brief",
    currentStep: "retrieval",
  });

  const response = await request(app).get(`/rag-estimations/${id}?step=brief`);

  assert.equal(response.status, 200);
  assert.match(response.text, /B2B payments platform/);
  assert.match(response.text, /Texto de búsqueda/);
});

test("POST /brief with save updates the brief without calling the LLM", async () => {
  let calls = 0;
  const { app, ragEstimations } = makeApp({
    ragClient: fakeRagClient({
      async brief() {
        calls += 1;
        return { brief: BRIEF, retrieval_text: "no" };
      },
    }),
  });
  const id = await seedRun(ragEstimations, { brief: BRIEF, retrievalText: "previo" });

  const response = await request(app)
    .post(`/rag-estimations/${id}/brief`)
    .type("form")
    .send({
      save: "1",
      functionality: "Nueva funcionalidad",
      technologies: "python, aws",
      sector: "healthcare",
      scale: "small",
      country: "FR",
      regulations: "GDPR",
      constraints: "3 months",
      retrieval_text: "nuevo texto",
    });

  assert.equal(response.status, 303);
  assert.equal(calls, 0);
  const run = await ragEstimations.getById(id);
  assert.equal(run.brief.functionality, "Nueva funcionalidad");
  assert.equal(run.retrievalText, "nuevo texto");
});

test("POST /retrieve stores the retrieval outcome", async () => {
  const { app, ragEstimations } = makeApp();
  const id = await seedRun(ragEstimations, {
    brief: BRIEF,
    retrievalText: "B2B payments",
    status: "brief",
    currentStep: "retrieval",
  });

  const response = await request(app)
    .post(`/rag-estimations/${id}/retrieve`)
    .type("form")
    .send({ query_text: "B2B payments", top_k: "5", distance_threshold: "0.5" });

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.retrieval.chunks.length, 1);
  assert.equal(run.status, "retrieved");
});

test("POST /assemble stores the context block", async () => {
  const { app, ragEstimations } = makeApp();
  const id = await seedRun(ragEstimations, {
    brief: BRIEF,
    retrievalText: "B2B",
    retrieval: RETRIEVAL,
    status: "retrieved",
    currentStep: "assembly",
  });

  const response = await request(app)
    .post(`/rag-estimations/${id}/assemble`)
    .type("form")
    .send({ max_context_tokens: "1000" });

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.assembly.token_count, 12);
  assert.equal(run.status, "assembled");
});

test("POST /generate stores the grounded estimate", async () => {
  const { app, ragEstimations } = makeApp();
  const id = await seedRun(ragEstimations, {
    brief: BRIEF,
    retrievalText: "B2B",
    retrieval: RETRIEVAL,
    assembly: ASSEMBLY,
    status: "assembled",
    currentStep: "generation",
  });

  const response = await request(app)
    .post(`/rag-estimations/${id}/generate`)
    .type("form")
    .send({ context_block: ASSEMBLY.context_block });

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.generation.estimate.total_engineer_days, 10);
  assert.equal(run.status, "generated");
});

test("POST /verify recomputes totals from the edited modules", async () => {
  const { app, ragEstimations } = makeApp();
  const id = await seedRun(ragEstimations, {
    brief: BRIEF,
    retrieval: RETRIEVAL,
    assembly: ASSEMBLY,
    generation: GENERATION,
    status: "generated",
    currentStep: "verification",
  });

  const body = [
    "modules[0][name]=Auth",
    "modules[0][description]=Login",
    "modules[0][tasks][0][name]=OAuth",
    "modules[0][tasks][0][engineer_days]=7",
    "modules[0][tasks][0][sources]=1,2",
    "modules[1][name]=QA",
    "modules[1][tasks][0][name]=Tests",
    "modules[1][tasks][0][engineer_days]=3",
    "duration_weeks=4",
    "notes=Recorte de alcance",
  ].join("&");

  const response = await request(app)
    .post(`/rag-estimations/${id}/verify`)
    .type("form")
    .send(body);

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.verification.total_engineer_days, 10);
  assert.equal(run.verification.duration_weeks, 4);
  assert.equal(run.status, "verified");
});

test("a client error keeps the run and shows the message", async () => {
  const { app, ragEstimations } = makeApp({
    ragClient: fakeRagClient({
      async brief() {
        throw new RagAuthError("API key inválida", { status: 401 });
      },
    }),
  });

  const response = await request(app)
    .post("/rag-estimations")
    .type("form")
    .send({ transcript: TRANSCRIPT });

  assert.equal(response.status, 303);
  const run = (await ragEstimations.list())[0];
  assert.equal(run.status, "failed");
  assert.match(response.headers.location, /error=/);
});

test("GET /rag-estimations/:id returns 404 for an unknown run", async () => {
  const { app } = makeApp();

  const response = await request(app).get("/rag-estimations/desconocido");

  assert.equal(response.status, 404);
});
