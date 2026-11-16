// Load the repo-root .env (ADMIN_PASSWORD, BACKEND_URL, ...) so the frontend
// shares one config file with the backend. Must be imported before anything
// that reads process.env. Already-set environment variables win.
import path from "node:path";
import { fileURLToPath } from "node:url";

const envFile = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", ".env");
try {
  process.loadEnvFile(envFile); // Node >= 20.12
} catch {
  /* no .env (e.g. on Cloud Run) or older Node: rely on real env vars */
}
