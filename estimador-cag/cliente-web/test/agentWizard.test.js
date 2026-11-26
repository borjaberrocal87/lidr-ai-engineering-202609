import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryAgentProfilesRepo } from "../src/agentProfilesRepo.js";
import { AgentEstimationError } from "../src/agentEstimationClient.js";
import { InMemoryRagEstimationRepo } from "../src/ragEstimationRepo.js";

const TRANSCRIPT = "x".repeat(120);

const BRIEF = {
  functionality: "App de pedidos con backend y movil",
  technologies: ["FastAPI"],
  sector: "logistics",
  scale: "medium",
  country: "ES",
  regulations: [],
  constraints: [],
};

const TRACE = {
  steps: [
    {
      step: 1,
      reasoning_summary: "Reformulo la busqueda del modelo de pedidos.",
      tool: "derive_task_hours",
      tool_args: { module: "Backend", task: "Modelo de pedidos", neighbors: [] },
      observation: "Modelo de pedidos: 55h from 2 analogs",
    },
  ],
  model: "gpt-5",
  iterations: 2,
  stopped_reason: "completed",
};

const AGENT_GENERATION = {
  estimate: {
    total_engineer_days: null,
    modules: [
      {
        name: "Backend",
        description: "API",
        tasks: [{ name: "Modelo de pedidos", description: "CRUD", engineer_days: null }],
      },
    ],
    duration_weeks: null,
    sources: [],
    assumptions: [],
    confidence: "medium",
    reasoning: "Descomposicion del agente.",
    insufficient_context_explanation: null,
  },
  fabricated_source_ids: [],
  citation_report: null,
  coherent: true,
  agent_trace: TRACE,
};

const AGENT_HOURS = {
  tasks: [
    {
      module: "Backend",
      task: "Modelo de pedidos",
      estimated_hours: 55,
      reliability: 0.7,
      has_match: true,
      dispersion: 0.1,
      neighbors: [],
      sources: [],
    },
  ],
  agent_trace: TRACE,
};

function fakeRagClient() {
  return {
    async brief() {
      return { brief: BRIEF, retrieval_text: "resumen" };
    },
    async structure() {
      return AGENT_GENERATION;
    },
    async taskHours() {
      return AGENT_HOURS;
    },
  };
}

function fakeAgentClient(overrides = {}) {
  const calls = { structure: [], hours: [] };
  const client = {
    calls,
    async proposeStructure(brief, knobs) {
      calls.structure.push({ brief, knobs });
      return structuredClone(AGENT_GENERATION);
    },
    async estimateTaskHours(modules, knobs) {
      calls.hours.push({ modules, knobs });
      return structuredClone(AGENT_HOURS);
    },
    ...overrides,
  };
  return client;
}

async function seedRun(ragEstimations) {
  const id = await ragEstimations.create({ transcript: TRANSCRIPT });
  await ragEstimations.update(id, {
    brief: BRIEF,
    status: "brief",
    currentStep: "generation",
  });
  return id;
}

function makeApp({ agentClient } = {}) {
  const ragEstimations = new InMemoryRagEstimationRepo();
  const agentProfiles = new InMemoryAgentProfilesRepo();
  const app = createApp({
    repo: null,
    estimator: null,
    ingestion: null,
    chunkingRepo: null,
    embeddings: null,
    config: null,
    ragEstimations,
    ragClient: fakeRagClient(),
    agentClient: agentClient || fakeAgentClient(),
    agentProfiles,
    corpusIndex: null,
    corpusRuns: null,
  });
  return { app, ragEstimations, agentProfiles };
}

async function seedProfile(agentProfiles, config = {}) {
  return agentProfiles.create({
    name: "Integraciones",
    agentType: "handwritten",
    persona: "Estimador senior.",
    config,
    isDefault: false,
  });
}

test("POST /generate-agent stores the agent generation and the chosen profile", async () => {
  const agentClient = fakeAgentClient();
  const { app, ragEstimations, agentProfiles } = makeApp({ agentClient });
  const id = await seedRun(ragEstimations);
  const profileId = await seedProfile(agentProfiles, { model: "gpt-5", search_top_k: 6 });

  const response = await request(app)
    .post(`/rag-estimations/${id}/generate-agent`)
    .type("form")
    .send(`profile_id=${profileId}`);

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.generation.agent.profileId, profileId);
  assert.equal(run.generation.agent.profileName, "Integraciones");
  assert.deepEqual(run.generation.agent.overrides, { model: "gpt-5", search_top_k: 6 });
  assert.equal(run.structure.modules[0].name, "Backend");
  assert.equal(agentClient.calls.structure[0].knobs.search_top_k, 6);
  assert.match(response.headers.location, /step=review/);
});

test("GET show renders the profile selector and the agent buttons", async () => {
  const { app, ragEstimations, agentProfiles } = makeApp();
  const id = await seedRun(ragEstimations);
  await seedProfile(agentProfiles, {});

  const generation = await request(app).get(`/rag-estimations/${id}?step=generation`);
  assert.equal(generation.status, 200);
  assert.match(generation.text, /Generar con agente/);
  assert.match(generation.text, /Integraciones/);

  await ragEstimations.update(id, {
    generation: AGENT_GENERATION,
    structure: { modules: AGENT_GENERATION.estimate.modules },
    status: "generated",
    currentStep: "review",
  });
  const review = await request(app).get(`/rag-estimations/${id}?step=review`);
  assert.equal(review.status, 200);
  assert.match(review.text, /Estimar horas con el agente/);
});

test("POST /estimate-hours-agent applies the stored overrides and renders the trace", async () => {
  const agentClient = fakeAgentClient();
  const { app, ragEstimations } = makeApp({ agentClient });
  const id = await seedRun(ragEstimations);
  await ragEstimations.update(id, {
    generation: {
      ...AGENT_GENERATION,
      agent: { profileId: "p1", profileName: "Integraciones", overrides: { model: "gpt-5" } },
    },
    structure: { modules: AGENT_GENERATION.estimate.modules },
    status: "generated",
    currentStep: "review",
  });

  const body = [
    "modules[0][name]=Backend",
    "modules[0][tasks][0][name]=Modelo de pedidos",
    "modules[0][tasks][0][description]=CRUD",
  ].join("&");

  const response = await request(app)
    .post(`/rag-estimations/${id}/estimate-hours-agent`)
    .type("form")
    .send(body);

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.taskHours.tasks[0].estimated_hours, 55);
  assert.equal(run.taskHours.agent_trace.steps.length, 1);
  assert.equal(agentClient.calls.hours[0].knobs.model, "gpt-5");
  assert.equal(run.verification.total_hours, 55);

  const hours = await request(app).get(`/rag-estimations/${id}?step=hours`);
  assert.equal(hours.status, 200);
  assert.match(hours.text, /Traza del agente/);
  assert.match(hours.text, /STEP 1/);
});

test("an agent client error keeps the run and redirects with the message", async () => {
  const agentClient = fakeAgentClient({
    async proposeStructure() {
      throw new AgentEstimationError("El agente no esta configurado.", { status: 500 });
    },
  });
  const { app, ragEstimations } = makeApp({ agentClient });
  const id = await seedRun(ragEstimations);

  const response = await request(app)
    .post(`/rag-estimations/${id}/generate-agent`)
    .type("form")
    .send("");

  assert.equal(response.status, 303);
  assert.match(response.headers.location, /error=/);
  assert.match(decodeURIComponent(response.headers.location), /no esta configurado/);
  const run = await ragEstimations.getById(id);
  assert.equal(run.status, "failed");
});

async function seedDefaultProfile(agentProfiles, config = {}) {
  return agentProfiles.create({
    name: "Rápido",
    agentType: "handwritten",
    persona: "",
    config,
    isDefault: true,
  });
}

test("POST /generate-agent falls back to the default profile when none is sent", async () => {
  const agentClient = fakeAgentClient();
  const { app, ragEstimations, agentProfiles } = makeApp({ agentClient });
  const id = await seedRun(ragEstimations);
  const profileId = await seedDefaultProfile(agentProfiles, {
    model: "gpt-5-mini",
    max_iterations: 6,
  });

  const response = await request(app)
    .post(`/rag-estimations/${id}/generate-agent`)
    .type("form")
    .send("");

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.generation.agent.profileId, profileId);
  assert.equal(run.generation.agent.profileName, "Rápido");
  assert.deepEqual(agentClient.calls.structure[0].knobs, {
    model: "gpt-5-mini",
    max_iterations: 6,
  });
});

test("an explicit empty profile_id opts out of the default profile", async () => {
  const agentClient = fakeAgentClient();
  const { app, ragEstimations, agentProfiles } = makeApp({ agentClient });
  const id = await seedRun(ragEstimations);
  await seedDefaultProfile(agentProfiles, { model: "gpt-5-mini" });

  const response = await request(app)
    .post(`/rag-estimations/${id}/generate-agent`)
    .type("form")
    .send("profile_id=");

  assert.equal(response.status, 303);
  const run = await ragEstimations.getById(id);
  assert.equal(run.generation.agent.profileId, null);
  assert.equal(agentClient.calls.structure[0].knobs, null);
});

test("hours without a stored agent use the default profile overrides", async () => {
  const agentClient = fakeAgentClient();
  const { app, ragEstimations, agentProfiles } = makeApp({ agentClient });
  const id = await seedRun(ragEstimations);
  await seedDefaultProfile(agentProfiles, { model: "gpt-5-mini", search_top_k: 7 });
  await ragEstimations.update(id, {
    generation: AGENT_GENERATION,
    structure: { modules: AGENT_GENERATION.estimate.modules },
    status: "generated",
    currentStep: "review",
  });

  const body = [
    "modules[0][name]=Backend",
    "modules[0][tasks][0][name]=Modelo de pedidos",
  ].join("&");

  const response = await request(app)
    .post(`/rag-estimations/${id}/estimate-hours-agent`)
    .type("form")
    .send(body);

  assert.equal(response.status, 303);
  assert.deepEqual(agentClient.calls.hours[0].knobs, {
    model: "gpt-5-mini",
    search_top_k: 7,
  });
});

test("the generation step preselects the default profile", async () => {
  const { app, ragEstimations, agentProfiles } = makeApp();
  const id = await seedRun(ragEstimations);
  const profileId = await seedDefaultProfile(agentProfiles, {});

  const response = await request(app).get(`/rag-estimations/${id}?step=generation`);

  assert.equal(response.status, 200);
  const selected = new RegExp(`value="${profileId}"[\\s\\S]{0,120}selected`);
  assert.match(response.text, selected);
});
