import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tests run fully offline and hermetically: hashing embedder, no reranker,
# no Gemini calls, and a throw-away index directory.
_TMP = Path(tempfile.mkdtemp(prefix="rag-tests-"))
os.environ["EMBED_BACKEND"] = "hash"
os.environ["RERANK_BACKEND"] = "none"
os.environ["GEMINI_API_KEY"] = ""
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
os.environ["QDRANT_URL"] = ""
os.environ["PROCESSED_DIR"] = str(_TMP / "processed")
os.environ["QDRANT_PATH"] = str(_TMP / "qdrant")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")


@pytest.fixture(scope="session", autouse=True)
def built_index():
    """Run Part 1 + Part 2 once for the whole test session."""
    from scripts import build_index, ingest

    ingest.main()
    build_index.main()
    yield
