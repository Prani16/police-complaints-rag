"""Part 4 - evaluation: compare retrieval strategies on eval/questions.jsonl.

    python -m scripts.eval                 # retrieval metrics only (no Gemini cost)
    python -m scripts.eval --generate      # + end-to-end answers via Gemini

Retrieval metrics per strategy:
  Hit@1 / Hit@5 : an expected document is ranked 1st / in the top 5
  MRR           : mean reciprocal rank of the first expected document
Strategies: dense only, BM25 only, hybrid (RRF), hybrid + reranker.
Out-of-scope questions check that the relevance gate says "not found".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import pipeline  # noqa: E402
from rag.config import ROOT, settings  # noqa: E402
from rag.reranker import get_reranker  # noqa: E402
from rag.retriever import get_retriever  # noqa: E402


def rank_of(doc_ids: list[str], expected: list[str]) -> int | None:
    for i, d in enumerate(doc_ids, 1):
        if d in expected:
            return i
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {"hit@1": sum(r == 1 for r in ranks) / n,
            "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--generate", action="store_true", help="also run Gemini end to end")
    args = ap.parse_args()

    qs = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    in_scope = [q for q in qs if not q.get("out_of_scope")]
    retr, reranker = get_retriever(), get_reranker()
    by_id = retr.by_id

    ranks = {"dense": [], "bm25": [], "hybrid": [], "hybrid+rerank": []}
    misses = []
    t0 = time.time()
    for q in in_scope:
        exp = q["expected_doc"]
        ranks["dense"].append(rank_of([by_id[c]["doc_id"] for c in retr.dense(q["question"], 20)], exp))
        ranks["bm25"].append(rank_of([by_id[c]["doc_id"] for c in retr.sparse(q["question"], 20)], exp))
        cands = retr.retrieve(q["question"])
        ranks["hybrid"].append(rank_of([c.chunk["doc_id"] for c in cands], exp))
        scores = reranker.score(q["question"], [c.chunk["embed_text"] for c in cands])
        if scores is not None:
            order = [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]
            r = rank_of([c.chunk["doc_id"] for c in order], exp)
            ranks["hybrid+rerank"].append(r)
            if r != 1:
                misses.append((q["id"], q["question"], order[0].chunk["doc_id"]))
    if not ranks["hybrid+rerank"]:
        del ranks["hybrid+rerank"]

    print(f"\nRetrieval on {len(in_scope)} in-scope questions  "
          f"(embedder={retr.embedder.name}, reranker={reranker.name}, {time.time() - t0:.1f}s)\n")
    print(f"{'strategy':16} {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7}")
    for name, rs in ranks.items():
        m = metrics(rs)
        print(f"{name:16} {m['hit@1']:7.2f} {m['hit@5']:7.2f} {m['mrr']:7.3f}")
    if misses:
        print("\nNot ranked first after reranking:")
        for qid, qtext, got in misses:
            print(f"  {qid}: {qtext[:70]}  -> top doc {got}")

    oos = [q for q in qs if q.get("out_of_scope")]
    if oos and scores is not None:
        gated = sum(not pipeline.answer(q["question"])["found"] for q in oos)
        print(f"\nOut-of-scope questions correctly refused: {gated}/{len(oos)} "
              f"(MIN_RELEVANCE={settings.min_relevance})")

    if args.generate:
        print(f"\nEnd-to-end with {settings.gemini_model}:")
        ok = 0
        for q in in_scope:
            out = pipeline.answer(q["question"])
            hit = all(s.lower() in out["answer"].lower() for s in q["must_contain"])
            cited = any(s["cited"] for s in out["sources"])
            ok += hit and cited
            print(f"  {q['id']} {'PASS' if hit and cited else 'FAIL'}  contains={hit} cited={cited}  "
                  f"{out['timings_ms']['generate_ms']:.0f}ms")
        print(f"answer accuracy (must_contain + citation): {ok}/{len(in_scope)}")
    retr.store.client.close()


if __name__ == "__main__":
    main()
