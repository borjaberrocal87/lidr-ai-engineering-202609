import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryAgentProfilesRepo } from "../src/agentProfilesRepo.js";
import {
  configPayload,
  normalizeProfileInput,
} from "../src/routes/agentProfiles.js";

const PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=";

function makeApp() {
  const agentProfiles = new InMemoryAgentProfilesRepo();
  const app = createApp({
    repo: null,
    estimator: null,
    ingestion: null,
    chunkingRepo: null,
    embeddings: null,
    config: null,
    ragEstimations: null,
    ragClient: null,
    agentClient: null,
    agentProfiles,
    corpusIndex: null,
    corpusRuns: null,
  });
  return { app, agentProfiles };
}

function form(overrides = {}) {
  return {
    name: "Integraciones ERP",
    agent_type: "handwritten",
    persona: "Eres un estimador senior de integraciones.",
    model: "gpt-5",
    reasoning_effort: "high",
    max_iterations: "8",
    search_top_k: "12",
    search_distance_threshold: "0.5",
    ...overrides,
  };
}

test("GET /agent-profiles starts empty", async () => {
  const { app } = makeApp();
  const response = await request(app).get("/agent-profiles");
  assert.equal(response.status, 200);
  assert.match(response.text, /Todavía no hay perfiles/);
});

test("POST /agent-profiles creates a profile with knobs and avatar", async () => {
  const { app, agentProfiles } = makeApp();

  const response = await request(app)
    .post("/agent-profiles")
    .type("form")
    .send(form({ avatar_data_url: PNG, is_default: "1" }));

  assert.equal(response.status, 303);
  const profiles = await agentProfiles.list();
  assert.equal(profiles.length, 1);
  assert.equal(profiles[0].config.model, "gpt-5");
  assert.equal(profiles[0].config.max_iterations, 8);
  assert.equal(profiles[0].config.search_distance_threshold, 0.5);
  assert.equal(profiles[0].isDefault, true);
  assert.equal(profiles[0].avatarContentType, "image/png");

  const index = await request(app).get("/agent-profiles");
  assert.match(index.text, /Integraciones ERP/);
  assert.match(index.text, /predeterminado/);

  const show = await request(app).get(`/agent-profiles/${profiles[0].id}`);
  assert.equal(show.status, 200);
  assert.match(show.text, /gpt-5/);
  assert.match(show.text, /search_top_k/);
  assert.match(show.text, /12/);
});

test("POST /agent-profiles rejects an invalid avatar and a bad effort", async () => {
  const { app } = makeApp();

  const badAvatar = await request(app)
    .post("/agent-profiles")
    .type("form")
    .send(form({ avatar_data_url: "data:text/plain;base64,aGVsbG8=" }));
  assert.equal(badAvatar.status, 422);
  assert.match(badAvatar.text, /PNG, JPEG, GIF o WebP/);

  const badEffort = await request(app)
    .post("/agent-profiles")
    .type("form")
    .send(form({ reasoning_effort: "turbo" }));
  assert.equal(badEffort.status, 422);
  assert.match(badEffort.text, /Esfuerzo de razonamiento no válido/);

  const oversized = await request(app)
    .post("/agent-profiles")
    .type("form")
    .send(form({ avatar_data_url: `data:image/png;base64,${"A".repeat(800000)}` }));
  assert.equal(oversized.status, 422);
  assert.match(oversized.text, /no puede superar/);
});

test("profiles names are unique per agent type", async () => {
  const { app } = makeApp();
  await request(app).post("/agent-profiles").type("form").send(form());

  const duplicate = await request(app)
    .post("/agent-profiles")
    .type("form")
    .send(form({ name: "integraciones erp" }));

  assert.equal(duplicate.status, 422);
  assert.match(duplicate.text, /Ya existe un perfil con ese nombre/);
});

test("editing a profile keeps the avatar when none is provided", async () => {
  const { app, agentProfiles } = makeApp();
  await request(app).post("/agent-profiles").type("form").send(form({ avatar_data_url: PNG }));
  const [profile] = await agentProfiles.list();

  const response = await request(app)
    .post(`/agent-profiles/${profile.id}`)
    .type("form")
    .send(form({ name: "ERP avanzado", persona: "", avatar_data_url: PNG }));

  assert.equal(response.status, 303);
  const updated = await agentProfiles.getById(profile.id);
  assert.equal(updated.name, "ERP avanzado");
  assert.equal(updated.persona, "");
  assert.equal(updated.avatarContentType, "image/png");
});

test("making a profile default clears the previous default", async () => {
  const { app, agentProfiles } = makeApp();
  await request(app).post("/agent-profiles").type("form").send(form({ is_default: "1" }));
  await request(app)
    .post("/agent-profiles")
    .type("form")
    .send(form({ name: "Otro perfil" }));
  const profiles = await agentProfiles.list();
  const other = profiles.find((profile) => profile.name === "Otro perfil");

  await request(app).post(`/agent-profiles/${other.id}/default`);

  const refreshed = await agentProfiles.list();
  assert.equal(refreshed.filter((profile) => profile.isDefault).length, 1);
  assert.equal(refreshed.find((profile) => profile.isDefault).id, other.id);
});

test("configPayload compacts unset knobs", () => {
  assert.deepEqual(
    configPayload({
      config: {
        model: "gpt-5",
        reasoning_effort: "",
        max_iterations: null,
        search_top_k: 8,
      },
    }),
    { model: "gpt-5", search_top_k: 8 },
  );
});

test("normalizeProfileInput reports range errors", () => {
  const { errors } = normalizeProfileInput(form({ max_iterations: "99" }));
  assert.equal(errors.length, 1);
  assert.match(errors[0], /Máximo de iteraciones debe estar entre 1 y 20/);
});
