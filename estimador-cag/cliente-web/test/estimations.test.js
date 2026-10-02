import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryEstimationRepo } from "../src/db.js";
import { GuardrailError, ProviderError, UnavailableError } from "../src/estimatorClient.js";

const RESULT = {
  summary: "Una landing page con CRM, blog y formulario de contacto.",
  confidence_pct: 70,
  phases: [
    {
      name: "Discovery",
      duration_weeks: 1,
      cost_eur: 2000,
      summary: "Alcance y entrevistas con marketing.",
    },
  ],
  total_duration_weeks: 1,
  total_cost_eur: 2000,
};

const RESPONSE = {
  result: RESULT,
  prompt_version: "v1",
  cached: false,
  model: "gpt-4o-mini",
  provider: "openai",
};

function fakeEstimator({ result, error } = {}) {
  return {
    async getContext() {
      return {
        available_versions: ["v1", "v2"],
        description_min_length: 20,
        description_max_length: 50000,
      };
    },
    async estimate() {
      if (error) throw error;
      return result ?? RESPONSE;
    },
  };
}

function makeApp(estimator) {
  return createApp({ repo: new InMemoryEstimationRepo(), estimator });
}

test("GET / renders the new estimation form", async () => {
  const response = await request(makeApp(fakeEstimator())).get("/");

  assert.equal(response.status, 200);
  assert.match(response.text, /Nueva estimación/);
  assert.match(response.text, /<form[^>]+action="\/estimations"/);
});

test("GET /health returns ok", async () => {
  const response = await request(makeApp(fakeEstimator())).get("/health");

  assert.equal(response.status, 200);
  assert.deepEqual(response.body, { status: "ok", service: "cliente-web" });
});

test("POST /estimations persists and redirects to the detail page", async () => {
  const app = makeApp(fakeEstimator());
  const response = await request(app).post("/estimations").type("form").send({
    description: "Plataforma de inventario para cinco tiendas con alertas y dashboard.",
    project_type: "web_saas",
    detail_level: "medium",
    output_format: "phases_table",
    prompt_version: "v1",
  });

  assert.equal(response.status, 303);
  assert.match(response.headers.location, /^\/estimations\/[0-9a-f-]+$/);

  const list = await request(app).get("/estimations");
  assert.match(list.text, /Plataforma de inventario/);

  const detail = await request(app).get(response.headers.location);
  assert.equal(detail.status, 200);
  assert.match(detail.text, /Una landing page con CRM/);
  assert.match(detail.text, /Discovery/);
});

test("POST /estimations shows a guardrail rejection", async () => {
  const app = makeApp(fakeEstimator({ error: new GuardrailError("Se detectó un email.", { reason: "pii" }) }));

  const response = await request(app).post("/estimations").type("form").send({
    description: "Contacta con ana@example.com para el proyecto de inventario.",
    project_type: "web_saas",
    detail_level: "medium",
    output_format: "phases_table",
    prompt_version: "v1",
  });

  assert.equal(response.status, 400);
  assert.match(response.text, /Se detectó un email/);
});

test("POST /estimations shows a provider error", async () => {
  const app = makeApp(fakeEstimator({ error: new ProviderError("Proveedor caído.") }));

  const response = await request(app).post("/estimations").type("form").send({
    description: "Plataforma de inventario para cinco tiendas con alertas y dashboard.",
    project_type: "web_saas",
    detail_level: "medium",
    output_format: "phases_table",
    prompt_version: "v1",
  });

  assert.equal(response.status, 502);
  assert.match(response.text, /Proveedor caído/);
});

test("POST /estimations rejects a short description", async () => {
  const app = makeApp(fakeEstimator());

  const response = await request(app)
    .post("/estimations")
    .type("form")
    .send({ description: "corto", project_type: "web_saas" });

  assert.equal(response.status, 422);
});

test("GET /estimations/:id returns 404 for an unknown id", async () => {
  const response = await request(makeApp(fakeEstimator())).get("/estimations/desconocido");

  assert.equal(response.status, 404);
});

test("POST /estimations surfaces an unreachable API", async () => {
  const app = makeApp(fakeEstimator({ error: new UnavailableError("No se pudo contactar.") }));

  const response = await request(app).post("/estimations").type("form").send({
    description: "Plataforma de inventario para cinco tiendas con alertas y dashboard.",
    project_type: "web_saas",
    detail_level: "medium",
    output_format: "phases_table",
    prompt_version: "v1",
  });

  assert.equal(response.status, 502);
  assert.match(response.text, /No se pudo contactar/);
});
