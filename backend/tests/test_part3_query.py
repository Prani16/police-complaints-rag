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
