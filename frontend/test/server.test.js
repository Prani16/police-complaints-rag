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
      if (req.url === "/api/admin/eval") {
        if (req.headers["x-admin-key"] !== "pw") { res.statusCode = 401; return res.end(JSON.stringify({ detail: "Wrong admin key." })); }
        return res.end(JSON.stringify({ latest: { summary: { "hit@k": 0.9 } }, history: [] }));
      }
      if (req.url === "/api/upload") {
        return res.end(JSON.stringify({ doc_id: "upload_x", name: decodeURIComponent(req.headers["x-filename"]),
          bytes: Buffer.byteLength(body), type: req.headers["content-type"] }));
      }
      if (req.url === "/api/documents") return res.end(JSON.stringify([{ doc_id: "upload_x", uploaded: true }]));
      if (req.url === "/api/documents/upload_x" && req.method === "DELETE") return res.end(JSON.stringify({ deleted: "upload_x" }));
      if (req.url === "/api/query") {
        const { question, scope } = JSON.parse(body);
        if (question.includes("scope?")) return res.end(JSON.stringify({ answer: `scope=${scope}`, sources: [], models: {}, timings_ms: {} }));
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
  process.env.ADMIN_PASSWORD = "pw";
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

test("forwards uploads as raw bytes with the encoded file name", async () => {
  const r = await fetch(`${base}/api/upload`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent("my notes (1).md") },
    body: "# Hello\nsome text",
  });
  assert.equal(r.status, 200);
  const d = await r.json();
  assert.equal(d.name, "my notes (1).md");
  assert.equal(d.bytes, 17);
  assert.equal(d.type, "application/octet-stream");
});

test("lists and deletes documents, passes the search scope", async () => {
  assert.equal((await (await fetch(`${base}/api/documents`)).json())[0].doc_id, "upload_x");
  assert.equal((await fetch(`${base}/api/documents/upload_x`, { method: "DELETE" })).status, 200);
  const d = await (await post("which scope?")).json();
  assert.equal(d.answer, "scope=all");
  const r = await fetch(`${base}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: "which scope?", scope: "uploads" }) });
  assert.equal((await r.json()).answer, "scope=uploads");
});

test("rejects uploads over the size limit with a JSON error", async () => {
  const big = Buffer.alloc(11 * 1024 * 1024, 97);
  const r = await fetch(`${base}/api/upload`, { method: "POST", headers: { "X-Filename": "big.txt" }, body: big });
  assert.equal(r.status, 413);
  assert.match((await r.json()).error, /larger than/);
});

const basic = (u, p) => ({ Authorization: `Basic ${Buffer.from(`${u}:${p}`).toString("base64")}` });

test("admin dashboard needs the admin password", async () => {
  const anon = await fetch(`${base}/admin/`);
  assert.equal(anon.status, 401);
  assert.match(anon.headers.get("www-authenticate"), /Basic/);
  assert.equal((await fetch(`${base}/admin/`, { headers: basic("admin", "wrong") })).status, 401);
  assert.equal((await fetch(`${base}/admin/api/eval`)).status, 401);
  const ok = await fetch(`${base}/admin/`, { headers: basic("admin", "pw") });
  assert.equal(ok.status, 200);
  assert.match(await ok.text(), /quality report/);
  const data = await (await fetch(`${base}/admin/api/eval`, { headers: basic("admin", "pw") })).json();
  assert.equal(data.latest.summary["hit@k"], 0.9);
});

test("the user page does not link to the admin area", async () => {
  assert.doesNotMatch(await (await fetch(`${base}/`)).text(), /admin/i);
});
