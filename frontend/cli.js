#!/usr/bin/env node
// CLI frontend.
//   node cli.js "How do I file a Zero FIR?"           one question (all documents)
//   node cli.js --uploads "What does my file say?"     search only uploaded files
//   node cli.js upload path\to\notes.pdf               upload and index a file
//   node cli.js                                        interactive mode (type "exit" to quit)
//   BACKEND_URL=https://<cloud-run-url> node cli.js
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import "./env.js";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { ask, uploadFile, BACKEND_URL } from "./backend-client.js";

const c = {
  bold: (s) => `\x1b[1m${s}\x1b[0m`,
  dim: (s) => `\x1b[2m${s}\x1b[0m`,
  cyan: (s) => `\x1b[36m${s}\x1b[0m`,
  red: (s) => `\x1b[31m${s}\x1b[0m`,
};

let scope = "all";

async function run(question) {
  try {
    const r = await ask(question, undefined, scope);
    console.log(`\n${c.bold("Answer")}\n${r.answer}\n`);
    const cited = r.sources.filter((s) => s.cited);
    const list = cited.length ? cited : r.sources.slice(0, 3);
    if (list.length) {
      console.log(c.bold("Sources"));
      for (const s of list) {
        const where = [s.section, s.pages?.length ? `p. ${s.pages.join(", ")}` : ""].filter(Boolean).join(" | ");
        const score = s.rerank_score != null ? ` rerank=${s.rerank_score.toFixed(2)}` : "";
        console.log(`  ${c.cyan(`[${s.n}]`)} ${s.source}${where ? ` - ${where}` : ""}${c.dim(score)}`);
      }
    }
    const t = r.timings_ms;
    console.log(c.dim(`\n${r.models.llm} | retrieve ${t.retrieve_ms}ms, rerank ${t.rerank_ms}ms, generate ${t.generate_ms}ms\n`));
  } catch (e) {
    console.error(c.red(`Error: ${e.message} (backend: ${BACKEND_URL})`));
    process.exitCode = 1;
  }
}

let args = process.argv.slice(2);
if (args[0] === "--uploads") {
  scope = "uploads";
  args = args.slice(1);
}
const arg = args.join(" ").trim();
if (args[0] === "upload" && args[1]) {
  const file = args.slice(1).join(" ");
  try {
    const r = await uploadFile(await readFile(file), encodeURIComponent(path.basename(file)));
    console.log(`Added "${r.title}" (${r.chunks} chunks in ${r.seconds}s) as ${r.doc_id}`);
    console.log(c.dim('Ask about it with: node cli.js --uploads "your question"'));
  } catch (e) {
    console.error(c.red(`Upload failed: ${e.message}`));
    process.exitCode = 1;
  }
} else if (arg) {
  await run(arg);
} else {
  console.log(c.bold("Police Complaints Assistant") + c.dim(`  (backend: ${BACKEND_URL}, type "exit" to quit)`));
  const rl = readline.createInterface({ input: stdin, output: stdout });
  for (;;) {
    const q = (await rl.question(c.cyan("\n? "))).trim();
    if (!q) continue;
    if (["exit", "quit", "q"].includes(q.toLowerCase())) break;
    await run(q);
  }
  rl.close();
}
