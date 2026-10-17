import assert from "node:assert/strict";
import test from "node:test";

import { HttpIngestionClient, IngestionError } from "../src/ingestionClient.js";

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

const client = () => new HttpIngestionClient({ baseUrl: "http://api.test", timeoutMs: 1000 });

test("listSources returns the parsed catalog", async () => {
  global.fetch = async () =>
    jsonResponse(200, { version: "1.1.0", sources: [{ name: "presupuestos_json" }] });

  const body = await client().listSources();

  assert.equal(body.version, "1.1.0");
  assert.equal(body.sources.length, 1);
});

test("listSources forwards the decision filter", async () => {
  let calledUrl = "";
  global.fetch = async (url) => {
    calledUrl = url;
    return jsonResponse(200, { version: "1.1.0", sources: [] });
  };

  await client().listSources({ decision: "include" });

  assert.match(calledUrl, /decision=include/);
});

test("startRun returns the created job on 202", async () => {
  global.fetch = async () =>
    jsonResponse(202, { job_id: "abc", source_name: "presupuestos_json", status: "pending" });

  const body = await client().startRun("presupuestos_json");

  assert.equal(body.status, "pending");
  assert.equal(body.job_id, "abc");
});

test("startRun maps 400 (source_not_included) to IngestionError", async () => {
  global.fetch = async () =>
    jsonResponse(400, { detail: { reason: "source_not_included", decision: "exclude" } });

  await assert.rejects(client().startRun("rate_card_xlsx"), (error) => {
    assert.ok(error instanceof IngestionError);
    assert.equal(error.status, 400);
    return true;
  });
});

test("startRun maps 404 (unknown_source) to IngestionError", async () => {
  global.fetch = async () => jsonResponse(404, { detail: { reason: "unknown_source" } });

  await assert.rejects(client().startRun("nope"), (error) => {
    assert.ok(error instanceof IngestionError);
    assert.equal(error.status, 404);
    return true;
  });
});

test("getJob throws IngestionError on 404", async () => {
  global.fetch = async () => jsonResponse(404, { detail: "job not found" });

  await assert.rejects(client().getJob("missing"), (error) => {
    assert.ok(error instanceof IngestionError);
    assert.equal(error.status, 404);
    return true;
  });
});

test("listJobs paginates through query params", async () => {
  let calledUrl = "";
  global.fetch = async (url) => {
    calledUrl = url;
    return jsonResponse(200, { items: [], limit: 5, offset: 10 });
  };

  const body = await client().listJobs({ limit: 5, offset: 10 });

  assert.equal(body.limit, 5);
  assert.match(calledUrl, /limit=5/);
  assert.match(calledUrl, /offset=10/);
});
