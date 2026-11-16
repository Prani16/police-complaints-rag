"""Full RAG evaluation -> eval/results/latest.json (shown on the admin dashboard).

    pip install -r requirements-eval.txt        (once)
    python -m scripts.ragas_eval                 (run from backend/, uvicorn stopped)
    python -m scripts.ragas_eval --limit 5       quick smoke run
    python -m scripts.ragas_eval --no-ragas      retrieval metrics only (no judge LLM calls)

What it measures, on eval/ragas_testset.jsonl (questions with ground-truth
answers, relevant documents and evidence phrases):

Retrieval (deterministic, see rag/eval_metrics.py)
  Hit rate@k, MRR@k, Precision@k, Recall@k for 5 strategies:
  dense only, BM25 only, hybrid (RRF), hybrid + reranker, and the final
  context actually sent to Gemini (after the max-per-document rule).

RAGAS (LLM-judged, Gemini as the judge through its OpenAI-compatible API)
  context_precision  are the relevant passages ranked high? (vs reference)
  context_recall     do the passages contain everything the reference needs?
  faithfulness       are the answer's claims supported by the passages?
  answer_relevancy   does the answer address the question?
  answer_correctness does the answer match the ground-truth reference?

Out-of-scope questions check that the system refuses instead of guessing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import pipeline  # noqa: E402
from rag.config import ROOT, settings  # noqa: E402
from rag.eval_metrics import mean_of, relevant_chunk_ids, retrieval_scores  # noqa: E402
from rag.reranker import get_reranker  # noqa: E402
from rag.retriever import get_retriever  # noqa: E402

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
RAGAS_METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
RETRIEVAL_METRICS = ["hit", "mrr", "precision", "recall"]


# --------------------------------------------------------------------------- RAGAS setup
def build_ragas(judge_model: str):
    """Gemini judge via its OpenAI-compatible endpoint (works with the RAGAS
    instructor adapter regardless of google-genai/instructor versions) and our
    own Qwen embedder for the two embedding-based metrics (no extra API)."""
    import instructor
    from openai import AsyncOpenAI
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms import llm_factory
    from ragas.metrics import collections as C

    client = AsyncOpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_URL, timeout=120)
    llm = llm_factory(judge_model, provider="openai", client=client,
                      mode=instructor.Mode.MD_JSON, temperature=0.0, max_tokens=4096)

    embedder = get_retriever().embedder

    class QwenEmbedding(BaseRagasEmbedding):
        def embed_text(self, text, **kw):
            return embedder.embed_documents([text])[0].tolist()

        async def aembed_text(self, text, **kw):
            return await asyncio.to_thread(self.embed_text, text)

    emb = QwenEmbedding()
    return {
        "context_precision": C.ContextPrecisionWithReference(llm=llm),
        "context_recall": C.ContextRecall(llm=llm),
        "faithfulness": C.Faithfulness(llm=llm),
        "answer_relevancy": C.AnswerRelevancy(llm=llm, embeddings=emb),
        "answer_correctness": C.AnswerCorrectness(llm=llm, embeddings=emb),
    }


async def run_metric(metric, sem: asyncio.Semaphore, **kwargs) -> tuple[float | None, str | None]:
    """Score one metric with retries on rate limits; never raises."""
    async with sem:
        for attempt in range(4):
            try:
                res = await asyncio.wait_for(metric.ascore(**kwargs), timeout=90)
                val = getattr(res, "value", res)
                return (None, "no score") if val is None else (round(float(val), 4), None)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                # retry only real rate limits / overload, never other errors
                transient = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "rate limit", "Rate limit",
                                                   "503", "UNAVAILABLE", "overloaded"))
                if transient and attempt < 3:
                    await asyncio.sleep(5 * 2 ** attempt)
                    continue
                return None, f"{type(exc).__name__}: {msg[:200]}"
    return None, "gave up"


async def preflight(judge_model: str, metrics: dict) -> str | None:
    """Check the judge once before the run: one plain call (key, model name,
    endpoint) and one tiny RAGAS metric (structured output). Returns an error
    message, or None if the judge works."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_URL, timeout=60)
    try:
        r = await client.chat.completions.create(
            model=judge_model, messages=[{"role": "user", "content": "Reply with the word OK."}], max_tokens=20)
        print(f"judge check 1/2: {judge_model} replied {r.choices[0].message.content!r}")
    except Exception as exc:  # noqa: BLE001
        return f"plain call to {judge_model} failed: {type(exc).__name__}: {str(exc)[:500]}"
    try:
        res = await asyncio.wait_for(metrics["faithfulness"].ascore(
            user_input="When must a Zero FIR be transferred?",
            response="A Zero FIR must be transferred within 24 hours.",
            retrieved_contexts=["The Zero FIR must be transferred within 24 hours through CCTNS."]), timeout=90)
        print(f"judge check 2/2: RAGAS faithfulness on a toy example = {getattr(res, 'value', res)}")
    except Exception as exc:  # noqa: BLE001
        return f"RAGAS metric call failed: {type(exc).__name__}: {str(exc)[:500]}"
    return None


async def ragas_for(q: dict, out: dict, metrics: dict, sem) -> tuple[dict, dict]:
    contexts = [s["text"] for s in out["sources"]]
    args = {
        "context_precision": dict(user_input=q["question"], reference=q["reference"], retrieved_contexts=contexts),
        "context_recall": dict(user_input=q["question"], retrieved_contexts=contexts, reference=q["reference"]),
        "faithfulness": dict(user_input=q["question"], response=out["answer"], retrieved_contexts=contexts),
        "answer_relevancy": dict(user_input=q["question"], response=out["answer"]),
        "answer_correctness": dict(user_input=q["question"], response=out["answer"], reference=q["reference"]),
    }
    scores, errors = {}, {}
    if not contexts:  # nothing retrieved/passed the gate: context metrics are 0, faithfulness n/a
        scores.update(context_precision=0.0, context_recall=0.0, faithfulness=None)
        errors["faithfulness"] = "no context (answer was refused)"
        args = {k: v for k, v in args.items() if k in ("answer_relevancy", "answer_correctness")}
    names = list(args)
    results = await asyncio.gather(*(run_metric(metrics[n], sem, **args[n]) for n in names))
    for n, (val, err) in zip(names, results):
        scores[n] = val
        if err:
            errors[n] = err
    return scores, errors


# --------------------------------------------------------------------------- retrieval strategies
def strategy_rankings(question: str, out: dict, k: int) -> dict[str, list[str]]:
    r, rr = get_retriever(), get_reranker()
    n = max(k, settings.rerank_candidates)
    cands = r.retrieve(question, k=n)
    ranked = {
        "dense": r.dense(question, n),
        "bm25": r.sparse(question, n),
        "hybrid": [c.chunk["chunk_id"] for c in cands],
    }
    scores = rr.score(question, [c.chunk["embed_text"] for c in cands]) if cands else None
    if scores is not None:
        order = sorted(zip(scores, cands), key=lambda x: -x[0])
        ranked["hybrid+rerank"] = [c.chunk["chunk_id"] for _, c in order]
    ranked["final_context"] = [s["chunk_id"] for s in out["sources"]]
    return ranked


# --------------------------------------------------------------------------- main
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


async def main_async(args) -> Path:
    qs = [json.loads(line) for line in open(args.testset, encoding="utf-8") if line.strip()]
    if args.limit:
        in_scope = [q for q in qs if not q.get("out_of_scope")][:args.limit]
        qs = in_scope + [q for q in qs if q.get("out_of_scope")][:max(1, args.limit // 4)]
    k = args.k
    retr = get_retriever()
    chunks = retr.chunks
    if any(c["doc_id"].startswith("upload_") for c in chunks):
        print("WARNING: uploaded documents are in the index; remove them for a clean evaluation.")

    import os

    judge = args.judge_model or os.getenv("RAGAS_JUDGE_MODEL") or settings.gemini_model
    use_ragas = not args.no_ragas
    skip_reason = None
    if use_ragas and not settings.gemini_api_key:
        use_ragas, skip_reason = False, "GEMINI_API_KEY not set (RAGAS needs Gemini as the judge)"
    metrics = None
    if use_ragas:
        try:
            metrics = build_ragas(judge)
        except ImportError as exc:
            use_ragas, skip_reason = False, f"RAGAS not installed ({exc}); pip install -r requirements-eval.txt"
    if use_ragas:
        problem = await preflight(judge, metrics)
        if problem:
            print(f"\nJUDGE NOT WORKING - stopping before the run.\n{problem}\n")
            print("Fix: check GEMINI_API_KEY, try --judge-model gemini-3.5-flash-lite, "
                  "or run with --no-ragas for retrieval metrics only.")
            raise SystemExit(1)
    elif args.no_ragas:
        skip_reason = "--no-ragas"
    sem = asyncio.Semaphore(args.concurrency)

    rows, strat_rows = [], defaultdict(list)
    t_start = time.time()
    for i, q in enumerate(qs, 1):
        t = time.perf_counter()
        try:
            out = pipeline.answer(q["question"])
        except RuntimeError as exc:  # Gemini error while answering
            out = {"answer": f"ERROR: {exc}", "found": False, "sources": [], "timings_ms": {}}
        latency = (time.perf_counter() - t) * 1000
        row = {"id": q["id"], "category": q.get("category", ""), "question": q["question"],
               "reference": q.get("reference", ""), "answer": out["answer"], "found": out.get("found", False),
               "latency_ms": round(latency), "timings_ms": out.get("timings_ms", {}),
               "contexts": [{"source": s["source"], "section": s["section"], "chunk_id": s["chunk_id"],
                             "text": s["text"], "rerank_score": s.get("rerank_score")} for s in out["sources"]],
               "out_of_scope": bool(q.get("out_of_scope"))}

        if q.get("out_of_scope"):
            row["refused"] = not out.get("found", False)
            print(f"[{i}/{len(qs)}] {q['id']} out-of-scope -> {'refused' if row['refused'] else 'ANSWERED'}")
            rows.append(row)
            continue

        rel = relevant_chunk_ids(q, chunks)
        row["relevant_chunks"] = sorted(rel)
        ranked = strategy_rankings(q["question"], out, k)
        for name, ids in ranked.items():
            sc = retrieval_scores(ids, rel, k)
            strat_rows[name].append(sc)
            if name == "final_context":
                row.update({f"{m}@k": v for m, v in sc.items()})
                row["relevant_flags"] = [cid in rel for cid in ids[:k]]

        if use_ragas:
            scores, errors = await ragas_for(q, out, metrics, sem)
            row.update(scores)
            if errors:
                row["errors"] = errors
        short = {"context_precision": "ctxP", "context_recall": "ctxR", "faithfulness": "faith",
                 "answer_relevancy": "relev", "answer_correctness": "correct"}
        shown = " ".join(f"{short[m]}={row.get(m)}" for m in RAGAS_METRICS) if use_ragas else ""
        print(f"[{i}/{len(qs)}] {q['id']} hit={row['hit@k']:.0f} mrr={row['mrr@k']:.2f} "
              f"p={row['precision@k']:.2f} r={row['recall@k']:.2f} {shown} ({latency:.0f} ms)")
        if row.get("errors"):
            first = next(iter(row["errors"].items()))
            print(f"      metric error ({first[0]}): {first[1][:300]}")
        rows.append(row)

    in_rows = [r for r in rows if not r["out_of_scope"]]
    oos_rows = [r for r in rows if r["out_of_scope"]]
    summary = {f"{m}@k": mean_of(in_rows, f"{m}@k") for m in RETRIEVAL_METRICS}
    summary.update({m: (mean_of(in_rows, m) if use_ragas else None) for m in RAGAS_METRICS})
    summary["refusal_rate"] = round(sum(r["refused"] for r in oos_rows) / len(oos_rows), 4) if oos_rows else None
    summary["false_refusal_rate"] = round(sum(not r["found"] for r in in_rows) / len(in_rows), 4) if in_rows else None
    summary["avg_latency_ms"] = round(sum(r["latency_ms"] for r in rows) / len(rows)) if rows else None

    by_cat = defaultdict(list)
    for r in in_rows:
        by_cat[r["category"]].append(r)
    categories = {c: {"n": len(rs), **{f"{m}@k": mean_of(rs, f"{m}@k") for m in RETRIEVAL_METRICS},
                      **{m: mean_of(rs, m) for m in RAGAS_METRICS}} for c, rs in sorted(by_cat.items())}
    strategies = {name: {m: mean_of(sc, m) for m in RETRIEVAL_METRICS} for name, sc in strat_rows.items()}

    gen = pipeline.get_generator()
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t_start, 1),
        "git_commit": git_commit(),
        "config": {
            "k": k, "testset": Path(args.testset).name, "n_questions": len(in_rows), "n_out_of_scope": len(oos_rows),
            "embedding": retr.embedder.name, "reranker": get_reranker().name,
            "llm": gen.model if gen.mode == "gemini" else "none (extractive)",
            "ragas": {"enabled": use_ragas, "judge_model": judge if use_ragas else None, "skipped_reason": skip_reason},
            "chunk_max_tokens": settings.chunk_max_tokens, "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "rerank_candidates": settings.rerank_candidates, "final_k": settings.final_k,
            "max_per_doc": settings.max_per_doc, "min_relevance": settings.min_relevance,
            "n_chunks": len(chunks),
        },
        "summary": summary, "strategies": strategies, "categories": categories, "questions": rows,
    }

    out_dir = Path(args.out_dir)
    (out_dir / "history").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    (out_dir / "history" / f"run-{stamp}.json").write_text(payload, encoding="utf-8")
    latest = out_dir / "latest.json"
    latest.write_text(payload, encoding="utf-8")

    print("\n==== summary ====")
    for key, val in summary.items():
        print(f"  {key:22} {val}")
    print("\nretrieval strategies (@%d):" % k)
    for name, m in strategies.items():
        print(f"  {name:15} " + "  ".join(f"{x}={m[x]}" for x in RETRIEVAL_METRICS))
    if skip_reason:
        print(f"\nRAGAS skipped: {skip_reason}")
    print(f"\nwrote {latest}  (open /admin in the app to view)")
    return latest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default=str(ROOT / "eval" / "ragas_testset.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--k", type=int, default=settings.final_k, help="cut-off for Hit/MRR/Precision/Recall")
    ap.add_argument("--limit", type=int, default=0, help="only the first N in-scope questions")
    ap.add_argument("--no-ragas", action="store_true", help="skip LLM-judged metrics")
    ap.add_argument("--judge-model", default="", help="Gemini model for RAGAS (default: GEMINI_MODEL)")
    ap.add_argument("--concurrency", type=int, default=3, help="parallel judge calls (lower if rate-limited)")
    args = ap.parse_args()
    asyncio.run(main_async(args))
    try:
        get_retriever().store.client.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
