"""Part 3b - rerankers. A reranker reads the query and a passage *together*
(cross-attention), which makes it more precise than the bi-encoder used for
first-stage retrieval. Pick one with RERANK_BACKEND:

  fast (default) cross-encoder/ms-marco-MiniLM-L6-v2 - 22M params, Apache-2.0.
                 ~25x smaller than Qwen3-Reranker-0.6B, so ~10 passages are
                 scored in well under a second on a laptop CPU.
  qwen           Qwen/Qwen3-Reranker-0.6B - best quality, decoder-based
                 P("yes"), but slow on CPU. Use on a GPU (or 4B/8B).
  bge            BAAI/bge-reranker-v2-m3 - multilingual cross-encoder.
  none           keep the RRF order (offline tests).

Every backend returns a relevance probability in [0, 1] so the "not found"
gate (MIN_RELEVANCE) works the same way for all of them.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import settings

log = logging.getLogger(__name__)

RERANK_INSTRUCTION = ("Given a question about police complaints, FIRs, arrest or custody "
                      "in India, judge whether the passage contains information that answers it")


class Reranker:
    name = "none"

    def score(self, query: str, passages: list[str]) -> list[float] | None:
        return None


class QwenReranker(Reranker):
    _PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
               "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
               "\"no\".<|im_end|>\n<|im_start|>user\n")
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading reranker %s", model_name)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        # float32 on CPU: the checkpoint is bf16, which is very slow on most CPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self._PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self._SUFFIX, add_special_tokens=False)
        self.max_length, self.batch_size, self.name = max_length, batch_size, model_name
        self._keep_last = True  # compute LM-head logits for the last token only

    def _last_logits(self, batch):
        # The LM head projects onto a ~150k-token vocabulary. Doing that for
        # every position wastes most of the compute; we only need the last one.
        if self._keep_last:
            try:
                return self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            except TypeError:  # older transformers without logits_to_keep
                self._keep_last = False
        return self.model(**batch).logits[:, -1, :]

    def score(self, query, passages):
        texts = [f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n<Document>: {p}" for p in passages]
        budget = self.max_length - len(self.pre) - len(self.suf)
        ids = [self.pre + x + self.suf for x in
               self.tok(texts, padding=False, truncation="longest_first", return_attention_mask=False,
                        max_length=budget)["input_ids"]]
        # batch similar lengths together so little time is spent on padding
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [0.0] * len(ids)
        with self.torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                batch = self.tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True,
                                     return_tensors="pt").to(self.device)
                logits = self._last_logits(batch)
                pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
                probs = self.torch.nn.functional.softmax(pair.float(), dim=1)[:, 1].tolist()
                for i, p in zip(idx, probs):
                    out[i] = p
        return out


class CrossEncoderReranker(Reranker):
    """Classic BERT-style cross-encoder (MiniLM, BGE, ...): one forward pass
    per (query, passage) pair, output passed through a sigmoid -> [0, 1]."""

    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16):
        import torch
        from sentence_transformers import CrossEncoder

        log.info("loading reranker %s", model_name)
        self.torch = torch
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.name, self.batch_size = model_name, batch_size

    def score(self, query, passages):
        return self.model.predict([(query, p) for p in passages], batch_size=self.batch_size,
                                  activation_fn=self.torch.nn.Sigmoid(),
                                  show_progress_bar=False).tolist()


DEFAULT_MODELS = {
    "fast": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "qwen": "Qwen/Qwen3-Reranker-0.6B",
    "bge": "BAAI/bge-reranker-v2-m3",
}


def model_for(backend: str, configured: str) -> str:
    """Use RERANK_MODEL only if it belongs to the chosen backend's family, so a
    leftover RERANK_MODEL=Qwen/... in .env can't be loaded as a cross-encoder."""
    is_qwen = "qwen3-reranker" in configured.lower()
    if configured and (is_qwen == (backend == "qwen")):
        return configured
    return DEFAULT_MODELS[backend]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    backend = settings.rerank_backend.lower()
    if backend not in DEFAULT_MODELS:
        return Reranker()
    name = model_for(backend, settings.rerank_model)
    if backend == "qwen":
        return QwenReranker(name, max_length=settings.rerank_max_tokens)
    return CrossEncoderReranker(name, max_length=settings.rerank_max_tokens)
