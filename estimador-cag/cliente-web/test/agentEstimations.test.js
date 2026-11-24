import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { AgentEstimationError } from "../src/agentEstimationClient.js";
import { createApp } from "../src/app.js";

const TRANSCRIPT = "x".repeat(120);

const RESULT = {
  report: {
    components: [
      {
        name: "SAP integration",
        estimated_hours: 299,
        source_chunk_ids: [105],
        rationale: "Grounded in an analogous ERP integration budget.",
      },
    ],
    total_hours: 299,
    assumptions: ["IDoc catalog is provided by the client."],
    confidence: "medium",
  },
  trace: {
    steps: [
      {
        step: 1,
        reasoning_summary: "The transcript names four components, start with the ERP one.",
        tool: "search_budgets",
        tool_args: { query: "SAP ERP integration", filters: null },
        observation: "2 historical items; hours=[260, 240]",
      },
    ],
    model: "gpt-5",
    iterations: 3,
    stopped_reason: "completed",
  },
  iterations: 3,
  stopped_reason: "completed",
  trace_text: "STEP 1\n  reasoning: ...\n  action: search_budgets({...})",
};

function fakeClient(overrides = {}) {
  return {
    estimateFromTranscript: async () => RESULT,
    ...overrides,
  };
}

function makeApp(agentClient) {
  return createApp({
    repo: null,
    estimator: null,
    ingestion: null,
    chunkingRepo: null,
    embeddings: null,
    config: null,
    ragEstimations: null,
    ragClient: null,
    agentClient,
    corpusIndex: null,
    corpusRuns: null,
  });
}

test("GET /agent-estimations/new renders the transcript form", async () => {
  const response = await request(makeApp(fakeClient())).get("/agent-estimations/new");
  assert.equal(response.status, 200);
  assert.match(response.text, /Agente de estimación/);
});

test("POST /agent-estimations rejects a short transcript", async () => {
  const response = await request(makeApp(fakeClient()))
    .post("/agent-estimations")
    .type("form")
    .send({ transcript: "too short" });
  assert.equal(response.status, 422);
  assert.match(response.text, /al menos 100 caracteres/);
});

test("POST /agent-estimations renders the trace and the report", async () => {
  let received = "";
  const client = fakeClient({
    estimateFromTranscript: async (transcript) => {
      received = transcript;
      return RESULT;
    },
  });
  const response = await request(makeApp(client))
    .post("/agent-estimations")
    .type("form")
    .send({ transcript: TRANSCRIPT });
  assert.equal(response.status, 200);
  assert.equal(received, TRANSCRIPT);
  assert.match(response.text, /SAP integration/);
  assert.match(response.text, /STEP 1/);
  assert.match(response.text, /Descargar traza/);
});

test("a client error is shown on the form with its message", async () => {
  const client = fakeClient({
    estimateFromTranscript: async () => {
      throw new AgentEstimationError("El agente no está configurado.", { status: 400 });
    },
  });
  const response = await request(makeApp(client))
    .post("/agent-estimations")
    .type("form")
    .send({ transcript: TRANSCRIPT });
  assert.equal(response.status, 400);
  assert.match(response.text, /no está configurado/);
});
