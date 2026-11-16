"""Part 3 tests: hybrid retrieval, relevance gate, citation handling, API."""
from fastapi.testclient import TestClient

from rag import pipeline
from rag.generator import NOT_FOUND, build_context
from rag.retriever import get_retriever, tokenize


def _top_doc(question: str) -> str:
    return get_retriever().retrieve(question)[0].chunk["doc_id"]


def test_tokenizer_keeps_legal_identifiers():
    assert {"173", "3", "41d"} <= set(tokenize("Section 173(3) and Section 41D BNSS"))


def test_hybrid_retrieval_finds_right_documents():
    assert _top_doc("How soon must a Zero FIR be transferred to the jurisdictional police station via CCTNS?").startswith("08")
    assert _top_doc("D.K. Basu arrest memo witness countersigned").startswith("06")
    assert _top_doc("within how many hours must NHRC be told about a custodial death").startswith("04")


def test_both_retrievers_contribute():
    cands = get_retriever().retrieve("preliminary enquiry 14 days DSP permission")
    assert any(c.dense_rank for c in cands) and any(c.sparse_rank for c in cands)


def test_pipeline_extractive_mode_without_gemini():
    out = pipeline.answer("What is a Zero FIR?")
    assert out["mode"] == "extractive"
    assert out["sources"] and out["sources"][0]["source"]
    assert set(out["timings_ms"]) == {"retrieve_ms", "rerank_ms", "generate_ms"}


def test_relevance_gate_skips_llm(monkeypatch):
    class LowReranker:
        name = "fake"

        def score(self, q, ps):
            return [0.01] * len(ps)

    class Boom:
        mode, model = "gemini", "fake"

        def generate(self, *a):
            raise AssertionError("LLM must not be called when nothing is relevant")

    monkeypatch.setattr(pipeline, "get_reranker", lambda: LowReranker())
    monkeypatch.setattr(pipeline, "get_generator", lambda: Boom())
    out = pipeline.answer("What is the capital of France?")
    assert out["answer"] == NOT_FOUND and out["sources"] == [] and not out["found"]


def test_citations_are_flagged(monkeypatch):
    class FakeGen:
        mode, model = "gemini", "fake-gemini"

        def generate(self, q, passages):
            assert "Source:" in build_context(passages)
            return "A Zero FIR can be filed at any police station [1]. It is transferred within 24 hours [2]."

    monkeypatch.setattr(pipeline, "get_generator", lambda: FakeGen())
    out = pipeline.answer("What is a Zero FIR?")
    flags = {s["n"]: s["cited"] for s in out["sources"]}
    assert flags[1] and flags[2] and not flags.get(3, False)


def test_api_endpoints():
    from api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        info = client.get("/api/info").json()
        assert len(info["documents"]) == 10
        r = client.post("/api/query", json={"question": "Who heads a district Police Complaints Authority?"})
        assert r.status_code == 200 and r.json()["sources"]
        assert client.post("/api/query", json={"question": "a"}).status_code == 422


def test_sources_are_diverse():
    from collections import Counter

    out = pipeline.answer("What can I do if the police refuse to register my FIR?")
    per_doc = Counter(s["source"] for s in out["sources"])
    assert len(out["sources"]) == 5 and max(per_doc.values()) <= 2


def test_diversify_fills_up_when_few_documents():
    from types import SimpleNamespace as N

    cands = [N(chunk={"doc_id": "a"}) for _ in range(4)] + [N(chunk={"doc_id": "b"})]
    got = pipeline.diversify(cands, 4, 2)
    assert [c.chunk["doc_id"] for c in got] == ["a", "a", "b", "a"]


def test_reranker_model_selection():
    from rag.reranker import model_for

    # a leftover Qwen model name must not be loaded by the fast cross-encoder
    assert model_for("fast", "Qwen/Qwen3-Reranker-0.6B") == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert model_for("fast", "") == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert model_for("fast", "cross-encoder/ms-marco-MiniLM-L12-v2") == "cross-encoder/ms-marco-MiniLM-L12-v2"
    assert model_for("qwen", "") == "Qwen/Qwen3-Reranker-0.6B"
    assert model_for("qwen", "Qwen/Qwen3-Reranker-4B") == "Qwen/Qwen3-Reranker-4B"
    assert model_for("bge", "cross-encoder/ms-marco-MiniLM-L6-v2") == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_cross_encoder_reranker_returns_probabilities(monkeypatch):
    import sys
    import types

    import numpy as np

    class FakeCE:
        def __init__(self, name, max_length):
            self.name, self.max_length = name, max_length

        def predict(self, pairs, batch_size, activation_fn, show_progress_bar):
            logits = np.array([len(p) / 10 - 3 for _, p in pairs])
            return activation_fn(logits)

    fake_torch = types.SimpleNamespace(nn=types.SimpleNamespace(Sigmoid=lambda: lambda x: 1 / (1 + np.exp(-x))))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCE))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from rag.reranker import CrossEncoderReranker

    r = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L6-v2", max_length=512)
    s = r.score("q", ["short", "a much longer and more relevant passage about FIRs"])
    assert len(s) == 2 and all(0 <= x <= 1 for x in s) and s[1] > s[0]
    assert r.model.max_length == 512
