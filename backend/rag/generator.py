"""Part 3c - grounded answer generation with Gemini.

Model choice (see docs/DESIGN_DECISIONS.md):
  * best quality, cost ignored : gemini-4-argon (flagship reasoning model)
  * demo / default             : gemini-3.5-flash-lite (cheapest current model)
Switch with the GEMINI_MODEL env var - no code change.

Auth: GEMINI_API_KEY (AI Studio key, stored in GCP Secret Manager on Cloud
Run) or Vertex AI with the service account (GOOGLE_GENAI_USE_VERTEXAI=true).
If neither is configured the pipeline still works in "extractive" mode and
returns the top passages, so the UI and tests never hard-fail.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from .config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful assistant that explains police complaint procedures in India
(FIR, Zero FIR, e-FIR, arrest safeguards, custodial deaths, Police Complaints Authorities, NHRC).

Rules:
1. Answer ONLY from the numbered context passages. Do not use outside knowledge.
2. After every factual sentence, cite the passage(s) it came from, like [1] or [2][4].
3. If the context does not contain the answer, reply exactly:
   "I couldn't find this in the provided documents." and nothing else.
4. Quote section numbers, time limits and case names exactly as written in the context.
5. Be concise: a short direct answer first, then up to 5 bullet points if steps or lists help.
6. Never invent section numbers, deadlines, phone numbers or case names.
7. This is general information, not legal advice; do not add disclaimers unless asked."""

NOT_FOUND = "I couldn't find this in the provided documents."


def build_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        loc = p["title"] + (f" > {p['section']}" if p.get("section") else "")
        if p.get("pages"):
            loc += f" (p. {', '.join(map(str, p['pages']))})"
        blocks.append(f"[{i}] Source: {p['source']} | {loc}\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


class Generator:
    def __init__(self):
        self.client = None
        self.mode = "extractive"
        if settings.use_vertex or settings.gemini_api_key:
            from google import genai

            if settings.use_vertex:
                self.client = genai.Client(vertexai=True, project=settings.gcp_project or None,
                                           location=settings.gcp_location)
            else:
                self.client = genai.Client(api_key=settings.gemini_api_key)
            self.mode = "gemini"
        self.model = settings.gemini_model

    def generate(self, question: str, passages: list[dict]) -> str:
        if not passages:
            return NOT_FOUND
        if self.client is None:
            return self._extractive(passages)
        from google.genai import types

        prompt = f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}\n\nAnswer:"
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                          temperature=settings.temperature, max_output_tokens=1024)
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                return (resp.text or "").strip() or NOT_FOUND
            except Exception as exc:  # noqa: BLE001 - surface any API error gracefully
                retryable = any(s in str(exc) for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if attempt < 2 and retryable:
                    time.sleep(2 ** attempt)
                    continue
                log.exception("Gemini call failed")
                raise RuntimeError(f"Gemini error ({self.model}): {exc}") from exc
        return NOT_FOUND

    @staticmethod
    def _extractive(passages: list[dict]) -> str:
        lines = ["Gemini is not configured (set GEMINI_API_KEY), so here are the most relevant passages:"]
        for i, p in enumerate(passages[:3], 1):
            snippet = p["text"].replace("\n", " ")
            lines.append(f"- {snippet[:300]}{'...' if len(snippet) > 300 else ''} [{i}]")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return Generator()
