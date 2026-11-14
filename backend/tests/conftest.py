import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tests run fully offline: no model downloads, no Gemini calls.
os.environ.setdefault("EMBED_BACKEND", "hash")
os.environ.setdefault("RERANK_BACKEND", "none")
os.environ.setdefault("GEMINI_API_KEY", "")
