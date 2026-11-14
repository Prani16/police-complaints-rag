"""Part 3b - rerankers.

Default: Qwen/Qwen3-Reranker-0.6B (Apache-2.0). It is a decoder-based
reranker: for each (query, passage) pair it reads both together and outputs
P("yes" | passage answers query). Cross-attention over the pair is what makes
a reranker more precise than the bi-encoder used for first-stage retrieval.

Alternatives wired in: "bge" -> BAAI/bge-reranker-v2-m3 cross-encoder,
"none" -> keep RRF order (used by offline tests).
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

    def __init__(self, model_name: str, max_length: int = 2048, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading reranker %s", model_name)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self._PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self._SUFFIX, add_special_tokens=False)
        self.max_length, self.batch_size, self.name = max_length, batch_size, model_name

    def score(self, query, passages):
        texts = [f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n<Document>: {p}" for p in passages]
        out: list[float] = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], padding=False, truncation="longest_first",
                               return_attention_mask=False,
                               max_length=self.max_length - len(self.pre) - len(self.suf))
                enc["input_ids"] = [self.pre + ids + self.suf for ids in enc["input_ids"]]
                batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**batch).logits[:, -1, :]
                pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
                out.extend(self.torch.nn.functional.softmax(pair.float(), dim=1)[:, 1].tolist())
        return out


class BGEReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        log.info("loading reranker %s", model_name)
        self.model, self.name = CrossEncoder(model_name, max_length=1024), model_name

    def score(self, query, passages):
        import torch

        return self.model.predict([(query, p) for p in passages],
                                  activation_fn=torch.nn.Sigmoid()).tolist()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    if settings.rerank_backend == "qwen":
        return QwenReranker(settings.rerank_model)
    if settings.rerank_backend == "bge":
        name = settings.rerank_model if "bge" in settings.rerank_model else "BAAI/bge-reranker-v2-m3"
        return BGEReranker(name)
    return Reranker()
