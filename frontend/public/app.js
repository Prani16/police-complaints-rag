const chat = document.getElementById("chat");
const form = document.getElementById("ask");
const input = document.getElementById("q");
const send = document.getElementById("send");
const tpl = document.getElementById("tplAnswer");

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

// Minimal, safe markdown: escape first, then bold, bullets and [n] citation chips.
function render(text) {
  const lines = esc(text).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*(?!\s)([^*]+?)\*(?=[\s).,;:]|$)/g, "$1<em>$2</em>");
    const m = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)$/);
    if (m) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${m[1]}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim()) html += `<p>${line}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html.replace(/\[(\d+)\]/g, '<a class="cite" href="#" data-n="$1">$1</a>');
}

function sourceCard(s, turnId) {
  const where = [s.section, s.pages?.length ? `p. ${s.pages.join(", ")}` : ""].filter(Boolean).join(" · ");
  const score = s.rerank_score != null ? `relevance ${(s.rerank_score * 100).toFixed(0)}%` : `rrf ${s.rrf_score}`;
  const el = document.createElement("details");
  el.className = `src${s.cited ? " cited" : ""}`;
  el.id = `${turnId}-src-${s.n}`;
  el.innerHTML = `
    <summary><span class="n">${s.n}</span><span class="type">${esc(s.doc_type)}</span>
      <span class="title">${esc(s.title)}${s.section ? `<span class="sec"> › ${esc(s.section)}</span>` : ""}</span>
      <span class="score">${esc(score)}</span></summary>
    <div class="where">${esc(s.source)}${where ? " · " + esc(where) : ""}</div>
    <pre>${esc(s.text)}</pre>`;
  return el;
}

let turns = 0;
async function askQuestion(question) {
  document.getElementById("empty")?.remove();
  const turnId = `t${++turns}`;
  const node = tpl.content.firstElementChild.cloneNode(true);
  node.querySelector(".q").textContent = question;
  const body = node.querySelector(".body");
  body.innerHTML = '<span class="thinking">Searching the documents…</span>';
  chat.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  send.disabled = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, scope: currentScope() }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    body.innerHTML = render(data.answer);
    if (!data.found) body.classList.add("notfound");
    const t = data.timings_ms || {};
    node.querySelector(".meta").textContent =
      `${data.models.llm} · rerank: ${data.models.reranker} · ${Math.round(t.total_ms ?? 0)} ms total ` +
      `(retrieve ${Math.round(t.retrieve_ms)} · rerank ${Math.round(t.rerank_ms)} · generate ${Math.round(t.generate_ms)})`;

    const box = node.querySelector(".sources");
    if (data.sources.length) {
      const h = document.createElement("h3");
      h.textContent = "Sources";
      box.appendChild(h);
      data.sources.forEach((s) => box.appendChild(sourceCard(s, turnId)));
    }
    body.querySelectorAll("a.cite").forEach((a) =>
      a.addEventListener("click", (e) => {
        e.preventDefault();
        const card = document.getElementById(`${turnId}-src-${a.dataset.n}`);
        if (card) { card.open = true; card.scrollIntoView({ behavior: "smooth", block: "center" }); card.classList.add("flash"); setTimeout(() => card.classList.remove("flash"), 1200); }
      }),
    );
  } catch (e) {
    body.innerHTML = `<p class="error">Something went wrong: ${esc(e.message)}</p>`;
  } finally {
    send.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q.length < 3) return;
  input.value = "";
  askQuestion(q);
});

document.querySelectorAll(".examples button").forEach((b) => b.addEventListener("click", () => askQuestion(b.textContent)));

const infoBtn = document.getElementById("infoBtn");
const panel = document.getElementById("infoPanel");
infoBtn.addEventListener("click", async () => {
  const open = panel.hidden;
  panel.hidden = !open;
  infoBtn.setAttribute("aria-expanded", String(open));
  infoBtn.textContent = open ? "Hide system info" : "About the system";
  if (!open || panel.dataset.loaded) return;
  try {
    const d = await (await fetch("/api/info")).json();
    panel.innerHTML = `
      <div><b>Embedding</b> ${esc(d.models.embedding)}</div>
      <div><b>Vector store</b> ${esc(d.vector_store)}</div>
      <div><b>Reranker</b> ${esc(d.models.reranker)}</div>
      <div><b>LLM</b> ${esc(d.models.llm)}</div>
      <div><b>Index</b> ${d.chunk_count} chunks from ${d.documents.length} documents</div>
      <ul>${d.documents.map((x) => `<li><span class="type">${esc(x.type)}</span> ${esc(x.title)} <small>(${x.chunks} chunks)</small></li>`).join("")}</ul>`;
    panel.dataset.loaded = "1";
  } catch (e) {
    panel.textContent = `Could not load system info: ${e.message}`;
  }
});

// ------------------------------------------------------------------ uploads
const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const statusEl = document.getElementById("uploadStatus");
const uploadsEl = document.getElementById("uploads");
const scopeUploads = document.getElementById("scopeUploads");
const ALLOWED = [".pdf", ".html", ".htm", ".md", ".markdown", ".txt"];
const MAX_MB = 10;

function currentScope() {
  return document.querySelector('input[name="scope"]:checked')?.value || "all";
}

function setScope(value) {
  const radio = document.querySelector(`input[name="scope"][value="${value}"]`);
  if (radio && !radio.disabled) radio.checked = true;
}

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

async function refreshUploads() {
  try {
    const docs = await (await fetch("/api/documents")).json();
    const ups = Array.isArray(docs) ? docs.filter((d) => d.uploaded) : [];
    uploadsEl.innerHTML = "";
    for (const d of ups) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="type">${esc(d.type)}</span><span class="name" title="${esc(d.source)}">${esc(d.title)}</span>
        <small>${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</small>
        <button type="button" class="rm" aria-label="Remove ${esc(d.title)}" title="Remove">×</button>`;
      li.querySelector(".rm").addEventListener("click", () => removeUpload(d));
      uploadsEl.appendChild(li);
    }
    scopeUploads.disabled = ups.length === 0;
    if (!ups.length) setScope("all");
    panel.dataset.loaded = "";  // "About the system" must re-fetch the document list
  } catch {
    /* backend not ready yet; the list stays empty */
  }
}

async function uploadFile(file) {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED.includes(ext)) return setStatus(`${file.name}: only PDF, HTML, Markdown or TXT files are supported.`, "err");
  if (file.size > MAX_MB * 1024 * 1024) return setStatus(`${file.name} is larger than ${MAX_MB} MB.`, "err");

  setStatus(`Reading, chunking and embedding ${file.name}… (large files can take a minute)`, "busy");
  drop.classList.add("busy");
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
    await refreshUploads();
    setScope("uploads");
    setStatus(`Added "${d.title}" (${d.chunks} chunk${d.chunks === 1 ? "" : "s"} in ${d.seconds}s). Search is now set to your uploads; ask a question below.`, "ok");
    input.focus();
  } catch (e) {
    setStatus(`Upload failed: ${e.message}`, "err");
  } finally {
    drop.classList.remove("busy");
    fileInput.value = "";
  }
}

async function removeUpload(d) {
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(d.doc_id)}`, { method: "DELETE" });
    if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
    setStatus(`Removed "${d.title}".`, "ok");
  } catch (e) {
    setStatus(`Could not remove ${d.title}: ${e.message}`, "err");
  }
  refreshUploads();
}

document.getElementById("uploadBtn").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => fileInput.files[0] && uploadFile(fileInput.files[0]));
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && uploadFile(e.dataTransfer.files[0]));

refreshUploads();
