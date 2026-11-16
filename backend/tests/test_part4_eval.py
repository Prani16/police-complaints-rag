"""Evaluation: retrieval metric maths, the RAGAS runner (with stand-in metrics,
no LLM calls), and the admin results endpoint."""
import asyncio
import json
from argparse import Namespace

from fastapi.testclient import TestClient

from rag.config import settings
from rag.eval_metrics import relevant_chunk_ids, retrieval_scores


def test_retrieval_scores_maths():
    rel = {"a", "c", "x"}  # x exists in the corpus but was not retrieved
    s = retrieval_scores(["b", "a", "c", "d", "e"], rel, k=5)
    assert s == {"hit": 1.0, "mrr": 0.5, "precision": 0.4, "recall": 2 / 3}
    assert retrieval_scores(["b", "d"], rel, k=5)["hit"] == 0.0
    assert retrieval_scores(["a"], rel, k=1) == {"hit": 1.0, "mrr": 1.0, "precision": 1.0, "recall": 1 / 3}


def test_relevance_needs_doc_and_evidence():
    q = {"relevant_docs": ["d1"], "evidence": ["14 days"]}
    chunks = [{"chunk_id": "1", "doc_id": "d1", "text": "completed within 14 Days"},
              {"chunk_id": "2", "doc_id": "d1", "text": "something else"},
              {"chunk_id": "3", "doc_id": "d2", "text": "also 14 days"}]
    assert relevant_chunk_ids(q, chunks) == {"1"}


def test_every_ground_truth_question_is_answerable_from_the_corpus():
    from rag.retriever import load_chunks

    chunks = load_chunks()
    for line in open(settings.raw_dir.parents[1] / "eval" / "ragas_testset.jsonl"):
        q = json.loads(line)
        if not q.get("out_of_scope"):
            assert q["reference"] and relevant_chunk_ids(q, chunks), q["id"]


def test_ragas_runner_writes_dashboard_json(tmp_path, monkeypatch):
    from scripts import ragas_eval

    class FakeMetric:
        def __init__(self, value):
            self.value = value

        async def ascore(self, **kw):
            assert kw["user_input"]
            if self.value == "boom":
                raise ValueError("judge returned junk")
            return type("R", (), {"value": self.value})()

    fake = {"context_precision": FakeMetric(0.9), "context_recall": FakeMetric(0.8), "faithfulness": FakeMetric(1.0),
            "answer_relevancy": FakeMetric(0.7), "answer_correctness": FakeMetric("boom")}
    monkeypatch.setattr(ragas_eval, "build_ragas", lambda judge: fake)

    async def judge_ok(judge, metrics):
        return None

    monkeypatch.setattr(ragas_eval, "preflight", judge_ok)
    object.__setattr__(settings, "gemini_api_key", "test-key")  # enables the RAGAS branch
    try:
        args = Namespace(testset=str(settings.raw_dir.parents[1] / "eval" / "ragas_testset.jsonl"),
                         out_dir=str(tmp_path), k=5, limit=4, no_ragas=False, judge_model="", concurrency=2)
        latest = asyncio.run(ragas_eval.main_async(args))
    finally:
        object.__setattr__(settings, "gemini_api_key", "")
    r = json.loads(latest.read_text())
    assert r["config"]["n_questions"] == 4 and r["config"]["ragas"]["enabled"]
    assert r["summary"]["faithfulness"] == 1.0 and r["summary"]["answer_correctness"] is None
    q = next(x for x in r["questions"] if not x["out_of_scope"])
    assert "judge returned junk" in q["errors"]["answer_correctness"]
    assert {"dense", "bm25", "hybrid", "final_context"} <= set(r["strategies"])
    assert len(list((tmp_path / "history").glob("run-*.json"))) == 1


def test_admin_eval_endpoint_requires_key(tmp_path):
    from api import app

    (tmp_path / "history").mkdir()
    (tmp_path / "latest.json").write_text(json.dumps({"summary": {"hit@k": 1}, "config": {}}))
    object.__setattr__(settings, "eval_results_dir", tmp_path)
    try:
        with TestClient(app) as c:
            object.__setattr__(settings, "admin_password", "")
            assert c.get("/api/admin/eval").status_code == 403          # admin disabled
            object.__setattr__(settings, "admin_password", "pw")
            assert c.get("/api/admin/eval").status_code == 401
            assert c.get("/api/admin/eval", headers={"X-Admin-Key": "nope"}).status_code == 401
            r = c.get("/api/admin/eval", headers={"X-Admin-Key": "pw"})
            assert r.status_code == 200 and r.json()["latest"]["summary"]["hit@k"] == 1
    finally:
        object.__setattr__(settings, "admin_password", "")


def test_broken_judge_stops_before_the_run(tmp_path, monkeypatch):
    import pytest

    from scripts import ragas_eval

    monkeypatch.setattr(ragas_eval, "build_ragas", lambda judge: {})

    async def judge_broken(judge, metrics):
        return "plain call failed: 404 model not found"

    monkeypatch.setattr(ragas_eval, "preflight", judge_broken)
    object.__setattr__(settings, "gemini_api_key", "test-key")
    try:
        args = Namespace(testset=str(settings.raw_dir.parents[1] / "eval" / "ragas_testset.jsonl"),
                         out_dir=str(tmp_path), k=5, limit=2, no_ragas=False, judge_model="", concurrency=1)
        with pytest.raises(SystemExit):
            asyncio.run(ragas_eval.main_async(args))
    finally:
        object.__setattr__(settings, "gemini_api_key", "")
    assert not (tmp_path / "latest.json").exists()
