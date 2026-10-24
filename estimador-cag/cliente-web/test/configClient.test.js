import assert from "node:assert/strict";
import test from "node:test";

import { HttpConfigClient, RuntimeConfigError } from "../src/configClient.js";

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

const client = () =>
  new HttpConfigClient({ baseUrl: "http://api.test", timeoutMs: 1000, cacheTtlMs: 60000 });

test("getModels reads the config endpoint", async () => {
  global.fetch = async () =>
    jsonResponse(200, { models: { PRIMARY_MODEL: { effective: "gpt-4o" } } });

  const payload = await client().getModels();

  assert.equal(payload.models.PRIMARY_MODEL.effective, "gpt-4o");
});

test("getPrimaryModel caches successful reads", async () => {
  let calls = 0;
  global.fetch = async () => {
    calls += 1;
    return jsonResponse(200, { models: { PRIMARY_MODEL: { effective: "gpt-4o" } } });
  };
  const config = client();

  assert.equal(await config.getPrimaryModel(), "gpt-4o");
  assert.equal(await config.getPrimaryModel(), "gpt-4o");
  assert.equal(calls, 1);
});

test("getPrimaryModel degrades to null when the API fails", async () => {
  global.fetch = async () => jsonResponse(503, { detail: "caído" });

  assert.equal(await client().getPrimaryModel(), null);
});

test("updateModels sends the partial payload", async () => {
  let body = null;
  global.fetch = async (url, options) => {
    body = JSON.parse(options.body);
    return jsonResponse(200, { models: {} });
  };

  await client().updateModels({ PRIMARY_MODEL: "gpt-4o", CRITIC_MODEL: null });

  assert.deepEqual(body, { models: { PRIMARY_MODEL: "gpt-4o", CRITIC_MODEL: null } });
});

test("updateModels maps a 422 to RuntimeConfigError", async () => {
  global.fetch = async () => jsonResponse(422, { detail: "clave desconocida" });

  await assert.rejects(client().updateModels({ PRIMARY_MODEL: "gpt-4o" }), (error) => {
    assert.ok(error instanceof RuntimeConfigError);
    assert.equal(error.status, 422);
    return true;
  });
});
