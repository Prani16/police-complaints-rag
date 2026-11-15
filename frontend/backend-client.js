// Small client for the Python RAG API, shared by the web server and the CLI.
//
// On Cloud Run the backend can be private (--no-allow-unauthenticated). Set
// BACKEND_AUTH=true and this client fetches a Google-signed ID token for the
// backend URL from the metadata server and sends it as a Bearer token.

const BACKEND_URL = (process.env.BACKEND_URL || "http://localhost:8080").replace(/\/$/, "");
const BACKEND_AUTH = process.env.BACKEND_AUTH === "true";
const TIMEOUT_MS = Number(process.env.BACKEND_TIMEOUT_MS || 300000);

let cachedToken = { value: null, expires: 0 };

async function idToken() {
  if (!BACKEND_AUTH) return null;
  if (cachedToken.value && Date.now() < cachedToken.expires) return cachedToken.value;
  const url =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity" +
    `?audience=${encodeURIComponent(BACKEND_URL)}`;
  const res = await fetch(url, { headers: { "Metadata-Flavor": "Google" } });
  if (!res.ok) throw new Error(`metadata server returned ${res.status}`);
  cachedToken = { value: await res.text(), expires: Date.now() + 50 * 60 * 1000 };
  return cachedToken.value;
}

async function call(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const token = await idToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${BACKEND_URL}${path}`, {
    ...options,
    headers,
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    const err = new Error(detail || `backend returned ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

export const ask = (question, topK) =>
  call("/api/query", { method: "POST", body: JSON.stringify({ question, top_k: topK }) });
export const info = () => call("/api/info");
export const health = () => call("/health");
export { BACKEND_URL };
