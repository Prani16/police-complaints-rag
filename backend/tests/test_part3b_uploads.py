"""Uploads: a new file becomes searchable immediately, can be searched on its
own, survives a restart, and can be deleted."""
from urllib.parse import quote

from fastapi.testclient import TestClient

from api import app
from rag.retriever import get_retriever, load_chunks

NOTE = """# Campus Security Office - Night Patrol Rules

## Patrol timings
The campus night patrol of the zebrafinch hostel starts at 22:30 and ends at 05:45.

## Reporting a theft on campus
Students must report a theft to the Campus Security Office within 12 hours using form CSO-7,
and may also register an FIR at the local police station.
"""


def _upload(client, name, body):
    return client.post("/api/upload", content=body.encode() if isinstance(body, str) else body,
                       headers={"X-Filename": quote(name), "Content-Type": "application/octet-stream"})


def test_upload_search_scope_and_delete():
    with TestClient(app) as client:
        r = _upload(client, "Campus Rules (v2).md", NOTE)
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["doc_id"] == "upload_Campus_Rules_v2" and info["chunks"] >= 1

        docs = client.get("/api/documents").json()
        assert any(d["doc_id"] == info["doc_id"] and d["uploaded"] for d in docs)
        assert sum(not d["uploaded"] for d in docs) == 10

        # whole collection: the unusual word only exists in the upload
        out = client.post("/api/query", json={"question": "When does the zebrafinch hostel patrol start?"}).json()
        assert out["sources"][0]["source"] == "upload_Campus_Rules_v2.md"

        # uploads-only scope: every source must come from the upload
        out = client.post("/api/query", json={"question": "How do I report a theft and register an FIR?",
                                               "scope": "uploads"}).json()
        assert out["sources"] and {s["source"] for s in out["sources"]} == {"upload_Campus_Rules_v2.md"}

        # persisted to chunks.jsonl, so it is still there after a restart
        assert any(c["doc_id"] == info["doc_id"] for c in load_chunks())

        # re-uploading the same name replaces instead of duplicating
        n_before = len(get_retriever().chunks)
        assert _upload(client, "Campus Rules (v2).md", NOTE).status_code == 200
        assert len(get_retriever().chunks) == n_before

        assert client.delete(f"/api/documents/{info['doc_id']}").status_code == 200
        assert not any(d["uploaded"] for d in client.get("/api/documents").json())
        assert not any(c["doc_id"] == info["doc_id"] for c in load_chunks())
        out = client.post("/api/query", json={"question": "zebrafinch hostel patrol"}).json()
        assert all(not s["source"].startswith("upload_") for s in out["sources"])


def test_upload_validation():
    with TestClient(app) as client:
        assert _upload(client, "virus.exe", "MZ...").status_code == 400
        assert _upload(client, "empty.txt", "").status_code == 400
        assert _upload(client, "scan.pdf", b"%PDF-1.4 not really a pdf").status_code == 400
        assert client.post("/api/upload", content=b"hello").status_code == 400          # no filename
        assert client.delete("/api/documents/01_lalita_kumari_fir_guidelines").status_code == 400  # built-in
        r = client.post("/api/query", json={"question": "anything here?", "scope": "uploads"})
        assert r.status_code == 400 and "upload" in r.json()["detail"]
