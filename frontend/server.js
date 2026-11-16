// Node.js web frontend: serves the chat UI and proxies questions to the
// Python RAG API. The browser only ever talks to this server, so the backend
// URL and auth stay server-side.
import "./env.js"; // must come first: loads ../.env into process.env
import crypto from "node:crypto";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ask, info, health, uploadFile, documents, deleteDocument, adminEval, BACKEND_URL } from "./backend-client.js";

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
    const scope = req.body?.scope === "uploads" ? "uploads" : "all";
    const result = await ask(question, req.body?.top_k, scope);
    result.timings_ms = { ...(result.timings_ms || {}), total_ms: Date.now() - started };
    res.json(result);
  } catch (e) {
    console.error("ask failed:", e.message);
    res.status(e.status || 502).json({ error: e.message });
  }
});

// ------------------------------------------------------------------ admin
// The evaluation dashboard lives at /admin. It is not linked from the user UI
// and needs HTTP Basic auth: user ADMIN_USER (default "admin"), password
// ADMIN_PASSWORD. Without ADMIN_PASSWORD the whole admin area is disabled.
const ADMIN_USER = process.env.ADMIN_USER || "admin";
const sameSecret = (a, b) => {
  const h = (x) => crypto.createHash("sha256").update(String(x)).digest();
  return crypto.timingSafeEqual(h(a), h(b));
};
function requireAdmin(req, res, next) {
  const pw = process.env.ADMIN_PASSWORD;
  if (!pw) return res.status(503).type("text").send("Admin dashboard is disabled. Set ADMIN_PASSWORD in .env and restart.");
  const [scheme, value] = (req.get("authorization") || "").split(" ");
  if (scheme === "Basic" && value) {
    const decoded = Buffer.from(value, "base64").toString("utf8");
    const i = decoded.indexOf(":");
    if (i > -1 && sameSecret(decoded.slice(0, i), ADMIN_USER) && sameSecret(decoded.slice(i + 1), pw)) return next();
  }
  res.set("WWW-Authenticate", 'Basic realm="RAG admin", charset="UTF-8"').status(401).type("text").send("Admin login required.");
}

app.get("/admin/api/eval", requireAdmin, async (_req, res) => {
  res.set("Cache-Control", "no-store");
  try {
    res.json(await adminEval());
  } catch (e) {
    res.status(e.status || 502).json({ error: e.message });
  }
});
app.use("/admin", requireAdmin, express.static(path.join(__dirname, "admin"), { index: "index.html" }));

const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 10);

app.post("/api/upload", express.raw({ type: () => true, limit: `${MAX_UPLOAD_MB}mb` }), async (req, res) => {
  const name = req.get("x-filename");
  if (!name) return res.status(400).json({ error: "Missing file name." });
  if (!req.body?.length) return res.status(400).json({ error: "The file is empty." });
  try {
    res.json(await uploadFile(req.body, name));
  } catch (e) {
    console.error("upload failed:", e.message);
    res.status(e.status || 502).json({ error: e.message });
  }
});

app.get("/api/documents", async (_req, res) => {
  try {
    res.json(await documents());
  } catch (e) {
    res.status(e.status || 502).json({ error: e.message });
  }
});

app.delete("/api/documents/:id", async (req, res) => {
  try {
    res.json(await deleteDocument(req.params.id));
  } catch (e) {
    res.status(e.status || 502).json({ error: e.message });
  }
});

// body-parser errors (e.g. file over the size limit) as JSON, not an HTML page
app.use((err, _req, res, _next) => {
  const tooBig = err.type === "entity.too.large";
  res.status(err.status || 500).json({ error: tooBig ? `File is larger than ${MAX_UPLOAD_MB} MB.` : err.message });
});

export default app;

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 3000);
  app.listen(port, () => console.log(`frontend on http://localhost:${port} -> backend ${BACKEND_URL}`));
}
