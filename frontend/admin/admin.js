// Admin evaluation dashboard. Reads /admin/api/eval (behind basic auth) and
// renders stat tiles, small-multiple bars, a trend line and score tables.
const $ = (id) => document.getElementById(id);
const tip = $("tip");
const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim();

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);

const RET = [
  ["hit@k", "Hit rate", "Share of questions with ≥1 relevant chunk in the top k"],
  ["mrr@k", "MRR", "Mean of 1 / rank of the first relevant chunk"],
  ["precision@k", "Precision@k", "Relevant chunks ÷ k. Capped by how many relevant chunks exist"],
  ["recall@k", "Recall@k", "Relevant chunks found ÷ all relevant chunks in the corpus"],
];
const RAGAS = [
  ["faithfulness", "Faithfulness", "Claims in the answer supported by the retrieved passages"],
  ["answer_relevancy", "Answer relevancy", "How directly the answer addresses the question"],
  ["answer_correctness", "Answer correctness", "Agreement with the ground-truth answer (facts + meaning)"],
  ["context_precision", "Context precision", "Are passages useful for the reference ranked near the top?"],
  ["context_recall", "Context recall", "Share of the reference's claims found in the passages"],
];
const STRATS = [
  ["dense", "Dense only"], ["bm25", "BM25 only"], ["hybrid", "Hybrid (RRF)"],
  ["hybrid+rerank", "Hybrid + rerank"], ["final_context", "Final context"],
];

// ------------------------------------------------------------------ tooltip
function showTip(e, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => (tip.hidden = true);
function hover(el, html) {
  el.addEventListener("mousemove", (e) => showTip(e, typeof html === "function" ? html() : html));
  el.addEventListener("mouseleave", hideTip);
}

// ------------------------------------------------------------------ sequential heat cell
const SEQ = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];
function luminance(rgb) {
  const m = rgb.match(/\d+/g)?.map(Number) || [255, 255, 255];
  const [r, g, b] = m.slice(0, 3).map((v) => ((v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function heat(td, v) {
  td.classList.add("heat");
  if (v == null) {
    td.classList.add("na");
    td.textContent = "–";
    return td;
  }
  td.textContent = fmt(v);
  td.style.background = `var(${SEQ[Math.min(6, Math.floor(v * 6.999))]})`;
  requestAnimationFrame(() => {
    const bg = getComputedStyle(td).backgroundColor;
    td.style.color = luminance(bg) < 0.25 ? "#ffffff" : css("--text-primary");
  });
  return td;
}

// ------------------------------------------------------------------ sections
function renderMeta(r) {
  const c = r.config;
  const when = new Date(r.created_at).toLocaleString();
  $("runMeta").textContent =
    `Run of ${when} · ${c.n_questions} questions + ${c.n_out_of_scope} out-of-scope · k = ${c.k} · ` +
    `${c.llm} · reranker ${c.reranker.split("/").pop()} · took ${r.duration_s}s` + (r.git_commit ? ` · commit ${r.git_commit}` : "");
  document.querySelectorAll("[data-k]").forEach((el) => (el.textContent = `k = ${c.k}`));
}

function tile(key, label, hint, v, opts = {}) {
  const d = document.createElement("div");
  d.className = `kpi${v == null ? " na" : ""}`;
  const shown = opts.ms ? (v == null ? "–" : `${(v / 1000).toFixed(1)}s`) : opts.pct ? pct(v) : fmt(v);
  d.innerHTML = `<div class="label">${esc(label)}</div><div class="value">${shown}</div>` +
    (opts.ms ? "" : `<div class="meter"><i style="width:${v == null ? 0 : Math.max(0, Math.min(1, v)) * 100}%"></i></div>`) +
    (opts.sub ? `<div class="hint">${esc(opts.sub)}</div>` : "");
  hover(d, `<b>${esc(label)}</b><br>${esc(hint)}`);
  return d;
}

function renderKpis(r) {
  const s = r.summary, box = $("kpis");
  box.innerHTML = "";
  const groups = [
    ["Retrieval (final context)", RET.map(([k, l, h]) => tile(k, l, h, s[k]))],
    ["Generation (RAGAS)", RAGAS.slice(0, 3).map(([k, l, h]) => tile(k, l, h, s[k]))],
    ["Context (RAGAS)", RAGAS.slice(3).map(([k, l, h]) => tile(k, l, h, s[k]))],
    ["Safety & speed", [
      tile("refusal_rate", "Refused off-topic", "Out-of-scope questions answered with 'not found'", s.refusal_rate, { pct: true, sub: "higher is better" }),
      tile("false_refusal_rate", "Wrongly refused", "In-scope questions that got 'not found'", s.false_refusal_rate, { pct: true, sub: "lower is better" }),
      tile("avg_latency_ms", "Avg. latency", "Mean time per question, retrieval to answer", s.avg_latency_ms, { ms: true }),
    ]],
  ];
  for (const [title, tiles] of groups) {
    const g = document.createElement("div");
    g.className = "kpi-group";
    g.innerHTML = `<h3>${esc(title)}</h3>`;
    const row = document.createElement("div");
    row.className = "kpi-row";
    tiles.forEach((t) => row.appendChild(t));
    g.appendChild(row);
    box.appendChild(g);
  }
}

function barRow(name, v, { strong = false, muted = false, tipHtml = "" } = {}) {
  const row = document.createElement("div");
  row.className = "hbar";
  row.innerHTML = `<div class="name${strong ? " strong" : ""}">${esc(name)}</div>
    <div class="track"><div class="bar${muted ? " muted" : ""}" style="width:${v == null ? 0 : Math.max(0, Math.min(1, v)) * 100}%"></div><div class="hit"></div></div>
    <div class="val${v == null ? " na" : ""}">${fmt(v)}</div>`;
  if (tipHtml) hover(row, tipHtml);
  return row;
}

function renderStrategies(r) {
  const box = $("strategies");
  box.innerHTML = "";
  const present = STRATS.filter(([k]) => r.strategies[k]);
  for (const [mk, label, hint] of RET) {
    const m = document.createElement("div");
    m.className = "mult";
    m.innerHTML = `<h3>${esc(label)}</h3>`;
    const best = Math.max(...present.map(([k]) => r.strategies[k][mk.replace("@k", "")] ?? -1));
    for (const [sk, sl] of present) {
      const v = r.strategies[sk][mk.replace("@k", "")];
      m.appendChild(barRow(sl, v, {
        strong: sk === "final_context",
        muted: v !== best,
        tipHtml: `<b>${esc(sl)}</b> · ${esc(label)}: <b>${fmt(v, 3)}</b>${v === best ? " (best)" : ""}<br>${esc(hint)}`,
      }));
    }
    box.appendChild(m);
  }
}

function renderRagas(r) {
  const box = $("ragasBars"), rag = r.config.ragas || {};
  box.innerHTML = "";
  $("ragasNote").textContent = rag.enabled
    ? `Judged by ${rag.judge_model} through RAGAS. Scores are 0–1, higher is better. When the judge is the same model that answered, scores can be slightly optimistic.`
    : `RAGAS was not run: ${rag.skipped_reason || "unknown reason"}. Run python -m scripts.ragas_eval with GEMINI_API_KEY set.`;
  for (const [k, l, h] of RAGAS) {
    const vals = r.questions.filter((q) => !q.out_of_scope && q[k] != null).map((q) => q[k]);
    const n = vals.length, lo = n ? Math.min(...vals) : null;
    box.appendChild(barRow(l, r.summary[k], {
      tipHtml: `<b>${esc(l)}</b>: <b>${fmt(r.summary[k], 3)}</b><br>${esc(h)}<br>${n} scored questions` + (n ? `, lowest ${fmt(lo)}` : ""),
    }));
  }
}

function renderTrend(history) {
  const runs = history.filter((h) => h.summary && h.summary["hit@k"] != null);
  if (runs.length < 2) return;
  $("trendSection").hidden = false;
  const series = [
    ["hit@k", "Hit rate", "--series-1", ""],
    ["answer_correctness", "Answer correctness", "--seq-600", "5 4"],
  ];
  const W = 900, H = 220, P = { l: 40, r: 190, t: 12, b: 28 };
  const x = (i) => P.l + (runs.length === 1 ? 0 : (i / (runs.length - 1)) * (W - P.l - P.r));
  const y = (v) => P.t + (1 - v) * (H - P.t - P.b);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Hit rate and answer correctness across runs">`;
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(g)}" y2="${y(g)}" stroke="var(--grid)" stroke-width="1"/>` +
      `<text x="${P.l - 8}" y="${y(g) + 4}" text-anchor="end" font-size="11" fill="var(--text-muted)">${g}</text>`;
  }
  runs.forEach((h, i) => {
    if (i === 0 || i === runs.length - 1 || runs.length <= 8) {
      const d = new Date(h.created_at);
      svg += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${d.getDate()}/${d.getMonth() + 1} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}</text>`;
    }
  });
  for (const [k, label, color, dash] of series) {
    const pts = runs.map((h, i) => [i, h.summary[k]]).filter(([, v]) => v != null);
    if (!pts.length) continue;
    svg += `<polyline fill="none" stroke="var(${color})" stroke-width="2" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ""}
      points="${pts.map(([i, v]) => `${x(i)},${y(v)}`).join(" ")}"/>`;
    for (const [i, v] of pts) svg += `<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="var(${color})" stroke="var(--surface-1)" stroke-width="2"/>`;
    const [li, lv] = pts[pts.length - 1];
    svg += `<text x="${x(li) + 10}" y="${y(lv) + 4}" font-size="12" fill="var(--text-secondary)">${esc(label)} ${fmt(lv)}</text>`;
  }
  runs.forEach((h, i) => {
    svg += `<rect class="hz" data-i="${i}" x="${x(i) - 12}" y="${P.t}" width="24" height="${H - P.t - P.b}" fill="transparent"/>`;
  });
  svg += "</svg>";
  const box = $("trend");
  box.innerHTML = `<div class="legend">${series.map(([, l, c, d]) =>
    `<span><i style="background:var(${c});${d ? "opacity:.75" : ""}"></i>${esc(l)}</span>`).join("")}</div>${svg}`;
  box.querySelectorAll("rect.hz").forEach((rect) => {
    const h = runs[Number(rect.dataset.i)];
    hover(rect, `<b>${esc(new Date(h.created_at).toLocaleString())}</b><br>` +
      series.map(([k, l]) => `${esc(l)}: <b>${fmt(h.summary[k])}</b>`).join("<br>") +
      `<br><span style="color:var(--text-muted)">${esc(h.config?.reranker || "")}</span>`);
  });
}

function renderCategories(r) {
  const t = $("catTable");
  const cols = [...RET.map(([k, l]) => [k, l]), ...RAGAS.map(([k, l]) => [k, l])];
  t.innerHTML = `<thead><tr><th>Type</th><th class="num">Questions</th>${cols.map(([, l]) => `<th class="num">${esc(l)}</th>`).join("")}</tr></thead>`;
  const tb = document.createElement("tbody");
  for (const [cat, row] of Object.entries(r.categories)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(cat)}</td><td class="num">${row.n}</td>`;
    for (const [k, l] of cols) {
      const td = heat(document.createElement("td"), row[k]);
      hover(td, `<b>${esc(cat)}</b> · ${esc(l)}: <b>${fmt(row[k], 3)}</b>`);
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  t.appendChild(tb);
}

function detailRow(q, span) {
  const tr = document.createElement("tr");
  tr.className = "detail";
  const flags = q.relevant_flags || [];
  const ctx = (q.contexts || []).map((c, i) =>
    `<li class="${flags[i] ? "rel" : ""}"><div class="src">[${i + 1}] ${esc(c.source)}${c.section ? " › " + esc(c.section) : ""}` +
    `${c.rerank_score != null ? ` · rerank ${fmt(c.rerank_score)}` : ""}${flags[i] ? " · <b>relevant</b>" : ""}</div>${esc(c.text.slice(0, 420))}${c.text.length > 420 ? "…" : ""}</li>`).join("");
  const errs = q.errors ? `<div class="errors">Metric errors: ${Object.entries(q.errors).map(([k, v]) => `${esc(k)}: ${esc(v)}`).join("; ")}</div>` : "";
  tr.innerHTML = `<td colspan="${span}"><div class="detail-grid">
      <div><h4>System answer</h4><pre>${esc(q.answer)}</pre></div>
      <div><h4>Ground truth</h4><p>${q.out_of_scope ? "<em>Out of scope: the system should refuse.</em>" : esc(q.reference)}</p></div>
    </div>
    <h4 style="margin-top:12px">Retrieved chunks (${(q.contexts || []).length})</h4><ul class="ctx">${ctx || "<li>None: the answer was refused before generation.</li>"}</ul>${errs}</td>`;
  return tr;
}

let RESULT = null;
function renderQuestions() {
  const r = RESULT, filter = $("qFilter").value, t = $("qTable");
  const cols = [["hit@k", "Hit"], ["mrr@k", "MRR"], ["precision@k", "P@k"], ["recall@k", "R@k"],
    ["context_precision", "Ctx P"], ["context_recall", "Ctx R"], ["faithfulness", "Faith."],
    ["answer_relevancy", "Relev."], ["answer_correctness", "Correct."]];
  const span = cols.length + 3;
  t.innerHTML = `<thead><tr><th>ID</th><th>Question</th>${cols.map(([, l]) => `<th class="num">${l}</th>`).join("")}<th class="num">Time</th></tr></thead>`;
  const tb = document.createElement("tbody");
  const rows = r.questions.filter((q) => {
    if (filter === "oos") return q.out_of_scope;
    if (q.out_of_scope) return filter === "all";
    if (filter === "weak") return (q.answer_correctness ?? 1) < 0.5 || (q.faithfulness ?? 1) < 0.7;
    if (filter === "miss") return q["hit@k"] === 0;
    return true;
  });
  for (const q of rows) {
    const tr = document.createElement("tr");
    tr.className = "q";
    tr.tabIndex = 0;
    tr.innerHTML = `<td>${esc(q.id)}</td><td class="qtext">${esc(q.question)}<div class="cat">${esc(q.category)}</div></td>`;
    if (q.out_of_scope) {
      const td = document.createElement("td");
      td.colSpan = cols.length;
      td.innerHTML = q.refused
        ? `<span class="badge good">✓ Refused correctly</span>`
        : `<span class="badge bad">✗ Answered an out-of-scope question</span>`;
      tr.appendChild(td);
    } else {
      for (const [k, l] of cols) {
        const td = heat(document.createElement("td"), q[k]);
        hover(td, `<b>${esc(q.id)}</b> · ${esc(l)}: <b>${fmt(q[k], 3)}</b>${q.errors?.[k] ? `<br>${esc(q.errors[k])}` : ""}`);
        tr.appendChild(td);
      }
    }
    const time = document.createElement("td");
    time.className = "num";
    time.textContent = `${(q.latency_ms / 1000).toFixed(1)}s`;
    tr.appendChild(time);
    let open = null;
    const toggle = () => {
      if (open) { open.remove(); open = null; return; }
      open = detailRow(q, span);
      tr.after(open);
    };
    tr.addEventListener("click", toggle);
    tr.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), toggle()));
    tb.appendChild(tr);
  }
  if (!rows.length) tb.innerHTML = `<tr><td colspan="${span}" style="color:var(--text-muted)">No questions match this filter.</td></tr>`;
  t.appendChild(tb);
}

function renderConfig(r) {
  const c = r.config, items = {
    "Embedding": c.embedding, "Reranker": c.reranker, "LLM": c.llm,
    "RAGAS judge": c.ragas?.judge_model || "not run", "Test set": c.testset, "Cut-off k": c.k,
    "Chunks in index": c.n_chunks, "Chunk size / overlap": `${c.chunk_max_tokens} / ${c.chunk_overlap_tokens} tokens`,
    "Rerank candidates": c.rerank_candidates, "Final passages": c.final_k, "Max per document": c.max_per_doc,
    "Relevance gate": c.min_relevance, "Git commit": r.git_commit || "–", "Run time": `${r.duration_s}s`,
  };
  $("config").innerHTML = Object.entries(items).map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
}

// ------------------------------------------------------------------ boot
async function main() {
  let data;
  try {
    const res = await fetch("/admin/api/eval", { cache: "no-store" });
    data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  } catch (e) {
    $("runMeta").textContent = "Could not load evaluation results.";
    $("empty").hidden = false;
    $("empty").innerHTML = `<p>${esc(e.message)}</p>`;
    return;
  }
  if (!data.latest) {
    $("runMeta").textContent = "No evaluation has been run yet.";
    $("empty").hidden = false;
    $("empty").innerHTML = `<p>Run the evaluation once, then reload this page:</p>
<pre>cd backend
pip install -r requirements-eval.txt
python -m scripts.ragas_eval</pre><p>Stop uvicorn first: the evaluation needs the local Qdrant folder to itself.</p>`;
    return;
  }
  RESULT = data.latest;
  $("dash").hidden = false;
  renderMeta(RESULT);
  renderKpis(RESULT);
  renderStrategies(RESULT);
  renderRagas(RESULT);
  renderTrend(data.history || []);
  renderCategories(RESULT);
  renderQuestions();
  renderConfig(RESULT);
  $("qFilter").addEventListener("change", renderQuestions);
}
main();
