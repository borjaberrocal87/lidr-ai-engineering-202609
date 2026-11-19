import assert from "node:assert/strict";
import test from "node:test";

import request from "supertest";

import { MODEL_KNOBS } from "../src/aiSettings.js";
import { createApp } from "../src/app.js";
import { InMemoryEstimationRepo } from "../src/db.js";

function catalog() {
  const models = {};
  for (const knob of MODEL_KNOBS) {
    models[knob.key] = {
      effective: "gpt-4o-mini",
      default: "gpt-4o-mini",
      overridden: false,
    };
  }
  return {
    models,
    available_models: ["gpt-4o-mini"],
    embedding_model: "text-embedding-3-small",
    embedding_model_note: "Solo lectura.",
  };
}

function retrievalSnapshot() {
  return {
    retrieval: {
      HALLUCINATION_GATE_ENABLED: { effective: true, default: true, overridden: false },
      AUGMENTATION_ENABLED: { effective: true, default: true, overridden: false },
      SYNTHESIS_ENABLED: { effective: false, default: true, overridden: true },
    },
    reranker_model: "cross-encoder",
  };
}

function fakeConfig() {
  const qualityUpdates = [];
  return {
    qualityUpdates,
    async getModels() {
      return catalog();
    },
    async getRetrieval() {
      return retrievalSnapshot();
    },
    async getPrimaryModel() {
      return "gpt-4o-mini";
    },
    async updateModels() {
      return catalog();
    },
    async updateRetrieval(changes) {
      qualityUpdates.push(changes);
      return retrievalSnapshot();
    },
  };
}

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

function makeApp(config) {
  return createApp({
    repo: new InMemoryEstimationRepo(),
    estimator: fakeEstimator(),
    ingestion: fakeIngestion(),
    config,
  });
}

test("GET /ai-settings renders the generation-quality toggles", async () => {
  const response = await request(makeApp(fakeConfig())).get("/ai-settings");

  assert.equal(response.status, 200);
  assert.match(response.text, /Calidad de generación/);
  assert.match(response.text, /Gate semántico/);
  assert.match(response.text, /Augmentación del contexto/);
  assert.match(response.text, /Síntesis de horas/);
});

test("POST /ai-settings/quality sends checkbox state as booleans", async () => {
  const config = fakeConfig();
  const response = await request(makeApp(config))
    .post("/ai-settings/quality")
    .type("form")
    .send({ hallucination_gate_enabled: "on", augmentation_enabled: "on" });

  assert.equal(response.status, 303);
  assert.equal(config.qualityUpdates.length, 1);
  assert.deepEqual(config.qualityUpdates[0], {
    hallucination_gate_enabled: true,
    augmentation_enabled: true,
    synthesis_enabled: false,
  });
});
