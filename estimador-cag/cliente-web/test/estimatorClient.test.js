import assert from "node:assert/strict";
import test from "node:test";

import {
  ConfigError,
  GuardrailError,
  HttpEstimatorClient,
  InputError,
  ProviderError,
  UnavailableError,
} from "../src/estimatorClient.js";

const originalFetch = global.fetch;

test.afterEach(() => {
  global.fetch = originalFetch;
});

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const client = () => new HttpEstimatorClient({ baseUrl: "http://api.test", timeoutMs: 1000 });

test("estimate returns the parsed body on 200", async () => {
  global.fetch = async () => jsonResponse(200, { result: { summary: "ok" }, prompt_version: "v1" });

  const body = await client().estimate({ description: "x" });

  assert.equal(body.prompt_version, "v1");
});

test("estimate maps 400 to GuardrailError with reason", async () => {
  global.fetch = async () =>
    jsonResponse(400, { detail: { reason: "pii", message: "Se detectó un email." } });

  await assert.rejects(client().estimate({ description: "x" }), (error) => {
    assert.ok(error instanceof GuardrailError);
    assert.equal(error.reason, "pii");
    return true;
  });
});

test("estimate maps 422 to InputError", async () => {
  global.fetch = async () => jsonResponse(422, { detail: "no válido" });

  await assert.rejects(client().estimate({ description: "x" }), InputError);
});

test("estimate maps 503 to ConfigError", async () => {
  global.fetch = async () => jsonResponse(503, { detail: "falta clave" });

  await assert.rejects(client().estimate({ description: "x" }), ConfigError);
});

test("estimate maps 502 to ProviderError", async () => {
  global.fetch = async () => jsonResponse(502, { detail: "proveedor" });

  await assert.rejects(client().estimate({ description: "x" }), ProviderError);
});

test("estimate maps a network failure to UnavailableError", async () => {
  global.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  await assert.rejects(client().estimate({ description: "x" }), UnavailableError);
});

test("getContext returns the payload", async () => {
  global.fetch = async () =>
    jsonResponse(200, { available_versions: ["v1", "v2"], description_min_length: 20 });

  const context = await client().getContext();

  assert.deepEqual(context.available_versions, ["v1", "v2"]);
});
