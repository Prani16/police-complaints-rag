// Node.js web frontend: serves the chat UI and proxies questions to the
// Python RAG API. The browser only ever talks to this server, so the backend
// URL and auth stay server-side.
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ask, info, health, BACKEND_URL } from "./backend-client.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: "16kb" }));
app.use(express.static(path.join(__dirname, "public")));

app.get("/healthz", async (_req, res) => {
  try {
    res.json({ frontend: "ok", backend: await health() });
  } catch (e) {
    res.status(503).json({ frontend: "ok", backend: "unreachable", error: e.message });
  }
});

app.get("/api/info", async (_req, res) => {
  try {
    res.json(await info());
  } catch (e) {
    res.status(e.status || 502).json({ error: e.message });
  }
});

app.post("/api/ask", async (req, res) => {
  const question = String(req.body?.question || "").trim();
  if (question.length < 3) return res.status(400).json({ error: "Please type a longer question." });
  if (question.length > 1000) return res.status(400).json({ error: "Question is too long (max 1000 characters)." });
  const started = Date.now();
  try {
    const result = await ask(question, req.body?.top_k);
    result.timings_ms = { ...(result.timings_ms || {}), total_ms: Date.now() - started };
    res.json(result);
  } catch (e) {
    console.error("ask failed:", e.message);
    res.status(e.status || 502).json({ error: e.message });
  }
});

export default app;

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 3000);
  app.listen(port, () => console.log(`frontend on http://localhost:${port} -> backend ${BACKEND_URL}`));
}
