import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { GraphEstimationError } from "../src/graphEstimationClient.js";

const TRANSCRIPT = "x".repeat(120);

const RESULT = {
  estimation_id: "est-1",
  status: "validated",
  estimate: {
    components: [
      {
        name: "Orders backend",
        estimated_hours: 126.5,
        unbudgeted: false,
        reference_budget_ids: ["B1"],
        rationale: "Median of 1 historical reference(s) plus a 15% contingency.",
      },
    ],
    total_hours: 126.5,
  },
  requirements: ["orders backend"],
  components: [{ name: "Orders backend", category: "backend" }],
  budget_matches: [
    { component: "Orders backend", reference_budget_id: "B1", amount: 110.0, distance: 0.1 },
  ],
  node_runs: [
    { name: "extract_requirements", duration_ms: 12.5 },
    { name: "validate_and_consolidate", duration_ms: 0.1 },
  ],
  errors: [],
  run_text: "SESSION 13 — ESTIMATION GRAPH RUN\nstatus: validated",
};

function fakeClient(overrides = {}) {
  return {
    estimateFromTranscript: async () => RESULT,
    ...overrides,
  };
}

function makeApp(graphClient) {
  return createApp({
    repo: null,
    estimator: null,
    ingestion: null,
    chunkingRepo: null,
    embeddings: null,
    config: null,
    ragEstimations: null,
    ragClient: null,
    agentClient: null,
    graphClient,
    agentProfiles: null,
    corpusIndex: null,
    corpusRuns: null,
  });
}

test("GET /graph-estimations/new renders the transcript form", async () => {
  const response = await request(makeApp(fakeClient())).get("/graph-estimations/new");
  assert.equal(response.status, 200);
  assert.match(response.text, /Grafo de estimación/);
});

test("POST /graph-estimations rejects a short transcript", async () => {
  const response = await request(makeApp(fakeClient()))
    .post("/graph-estimations")
    .type("form")
    .send({ transcript: "too short" });
  assert.equal(response.status, 422);
  assert.match(response.text, /al menos 100 caracteres/);
});

test("POST /graph-estimations renders the node timeline and the estimate", async () => {
  let received = "";
  const client = fakeClient({
    estimateFromTranscript: async (transcript) => {
      received = transcript;
      return RESULT;
    },
  });
  const response = await request(makeApp(client))
    .post("/graph-estimations")
    .type("form")
    .send({ transcript: TRANSCRIPT });
  assert.equal(response.status, 200);
  assert.equal(received, TRANSCRIPT);
  assert.match(response.text, /Orders backend/);
  assert.match(response.text, /extract_requirements/);
  assert.match(response.text, /validated/);
  assert.match(response.text, /126.5 h/);
  assert.match(response.text, /Descargar ejecución/);
});

test("a client error is shown on the form with its message", async () => {
  const client = fakeClient({
    estimateFromTranscript: async () => {
      throw new GraphEstimationError("El grafo no está disponible.", { status: 503 });
    },
  });
  const response = await request(makeApp(client))
    .post("/graph-estimations")
    .type("form")
    .send({ transcript: TRANSCRIPT });
  assert.equal(response.status, 503);
  assert.match(response.text, /no está disponible/);
});
