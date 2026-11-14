// Frontend tests: spin up a fake backend, point the Node server at it, and
// check the proxy, validation and error handling.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";

let fake, server, base;

before(async () => {
  fake = http.createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/api/query") {
        const { question } = JSON.parse(body);
        if (question.includes("boom")) { res.statusCode = 502; return res.end(JSON.stringify({ detail: "Gemini error" })); }
        return res.end(JSON.stringify({ answer: `echo: ${question} [1]`, found: true, sources: [{ n: 1, cited: true }],
          models: { llm: "fake" }, timings_ms: { retrieve_ms: 1, rerank_ms: 1, generate_ms: 1 } }));
      }
      if (req.url === "/health") return res.end(JSON.stringify({ status: "ok" }));
      res.statusCode = 404; res.end("{}");
    });
  });
  await new Promise((r) => fake.listen(0, r));
  process.env.BACKEND_URL = `http://127.0.0.1:${fake.address().port}`;
  const { default: app } = await import("../server.js");
  server = app.listen(0);
  await new Promise((r) => server.once("listening", r));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => { server.close(); fake.close(); });

const post = (q) => fetch(`${base}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });

test("proxies a question and adds total latency", async () => {
  const r = await post("What is a Zero FIR?");
  assert.equal(r.status, 200);
  const d = await r.json();
  assert.equal(d.answer, "echo: What is a Zero FIR? [1]");
  assert.ok(d.timings_ms.total_ms >= 0);
});

test("rejects too-short questions", async () => {
  assert.equal((await post("a")).status, 400);
});

test("surfaces backend errors", async () => {
  const r = await post("make it boom");
  assert.equal(r.status, 502);
  assert.match((await r.json()).error, /Gemini/);
});

test("serves the UI and health check", async () => {
  assert.equal((await fetch(`${base}/`)).status, 200);
  assert.equal((await (await fetch(`${base}/healthz`)).json()).backend.status, "ok");
});
