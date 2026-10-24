import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { createApp } from "../src/app.js";
import { InMemoryEstimationRepo } from "../src/db.js";
import { RuntimeConfigError } from "../src/configClient.js";

const CATALOG = {
  models: {
    PRIMARY_MODEL: { effective: "gpt-4o-mini", default: "gpt-4o-mini", overridden: false },
    FALLBACK_MODEL: { effective: "claude-haiku-4-5", default: "claude-haiku-4-5", overridden: false },
    CRITIC_MODEL: { effective: "gpt-4o-mini", default: "gpt-4o-mini", overridden: false },
    METADATA_EXTRACTOR_MODEL: {
      effective: "gpt-4o-mini",
      default: "gpt-4o-mini",
      overridden: false,
    },
    COMPRESSION_MODEL: { effective: "gpt-4o-mini", default: "gpt-4o-mini", overridden: false },
    PROPOSITIONAL_CHUNKER_MODEL: {
      effective: "gpt-4o-mini",
      default: "gpt-4o-mini",
      overridden: false,
    },
    CONTEXTUAL_CHUNKER_MODEL: {
      effective: "claude-sonnet-4-5",
      default: "claude-sonnet-4-5",
      overridden: false,
    },
  },
  available_models: ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5", "claude-sonnet-4-5"],
  embedding_model: "text-embedding-3-small",
  embedding_model_note: "Solo lectura.",
};

function fakeEstimator() {
  return {
    async getContext() {
      return { available_versions: ["v1"], description_min_length: 20, description_max_length: 100 };
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

function fakeConfig({ failGet = false, failUpdate = false } = {}) {
  const updates = [];
  return {
    updates,
    async getModels() {
      if (failGet) throw new RuntimeConfigError("API caída", { status: 503 });
      return CATALOG;
    },
    async getPrimaryModel() {
      return "gpt-4o-mini";
    },
    async updateModels(models) {
      if (failUpdate) throw new RuntimeConfigError("Cambio rechazado", { status: 422 });
      updates.push(models);
      return CATALOG;
    },
  };
}

function makeApp(config) {
  return createApp({
    repo: new InMemoryEstimationRepo(),
    estimator: fakeEstimator(),
    ingestion: fakeIngestion(),
    config,
  });
}

test("GET /ai-settings renders every knob", async () => {
  const response = await request(makeApp(fakeConfig())).get("/ai-settings");

  assert.equal(response.status, 200);
  assert.match(response.text, /Modelo primario/);
  assert.match(response.text, /Chunker contextual/);
  assert.match(response.text, /text-embedding-3-small/);
});

test("GET /ai-settings shows the config error", async () => {
  const response = await request(makeApp(fakeConfig({ failGet: true }))).get("/ai-settings");

  assert.equal(response.status, 200);
  assert.match(response.text, /API caída/);
});

test("POST /ai-settings sends empty selects as null", async () => {
  const config = fakeConfig();
  const response = await request(makeApp(config))
    .post("/ai-settings")
    .type("form")
    .send({ PRIMARY_MODEL: "gpt-4o", CRITIC_MODEL: "", FALLBACK_MODEL: "" });

  assert.equal(response.status, 303);
  assert.equal(config.updates.length, 1);
  assert.equal(config.updates[0].PRIMARY_MODEL, "gpt-4o");
  assert.equal(config.updates[0].CRITIC_MODEL, null);
  assert.equal(config.updates[0].FALLBACK_MODEL, null);
});

test("POST /ai-settings maps a validation error to 422", async () => {
  const response = await request(makeApp(fakeConfig({ failUpdate: true })))
    .post("/ai-settings")
    .type("form")
    .send({ PRIMARY_MODEL: "gpt-5" });

  assert.equal(response.status, 422);
  assert.match(response.text, /Cambio rechazado/);
});

test("the navbar shows the effective primary model", async () => {
  const response = await request(makeApp(fakeConfig())).get("/ai-settings");

  assert.match(response.text, /gpt-4o-mini/);
});
