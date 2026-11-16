# Design decisions and why

This document explains each component choice, the alternatives considered, and where each alternative falls short for this project. The constraints were:

- free, or as close to free as possible
- deployable on GCP
- Gemini as the LLM
- a Node.js frontend
- a small but messy corpus in four formats (PDF, HTML, Markdown, TXT)

Benchmark figures come from the sources linked at the end and were checked in November 2026. Leaderboards move every month, so re-check them before a real submission.

---

## 1. Chunking: structure-aware recursive chunking

**Choice.** The chunker works in these steps:

1. Split on document structure first: headings for MD and HTML, font-size headings and pages for PDF, ALL-CAPS headings for TXT. A chunk never crosses a section boundary.
2. Inside a section, pack whole paragraphs, list items and table rows up to **about 400 tokens**.
3. Fall back recursively to sentences, then words, only when one block is too big.
4. Add **about 60 tokens (15%) of sentence overlap** when a section spans several chunks.
5. Merge fragments under 80 tokens into a neighbour.
6. Prefix the embedded text with a **contextual header** (`Document title > Section`). The LLM-visible text stays clean.

| Strategy | Where it underperforms here |
|---|---|
| **Fixed-size** (every N characters or tokens) | Cuts mid-sentence and mid-list. For legal text this splits "Directive Six" across two chunks, or separates a deadline from the rule it belongs to. It was the weakest structured method in end-to-end benchmarks. |
| **Plain recursive character splitting** | A strong default: about 69% end-to-end accuracy at 512 tokens in the FloTorch 2026 benchmark, the best of 7 methods. But it ignores headings, so "14 days" can land in a chunk that never says "preliminary enquiry". The heading-first pass and contextual header fix exactly that. |
| **Semantic chunking** (split where embedding similarity drops) | Higher retrieval recall in isolation (about 92% vs 85–89% in Chroma's test). But it produced fragments averaging 43 tokens, which scored only 54% end-to-end answer accuracy. It also costs an embedding call per sentence at ingest. |
| **Page-level** | Won NVIDIA's benchmark for paginated financial PDFs. Our HTML, MD and TXT files have no pages, and our PDF sections span pages. |
| **Whole document / very large chunks** | Dilutes relevance. One embedding has to represent several topics, and the LLM gets lots of irrelevant text. |
| **Sentence-level / very small chunks** | Loses context. "It must be completed within 14 days" is useless without its subject. |

Independent benchmarks converge on 300–512 tokens with 10–20% overlap. Chroma's evaluation found that recursive splitting at 400–512 tokens with 10–20% overlap gave 85–90% Recall@5 with the lowest variance across document types. An arXiv study (2606.00881) concluded that *chunk quality and structural coherence matter more than chunk quantity*. That is the reason for splitting on structure first.

**Result on this corpus:** 40 chunks, averaging about 185 tokens, and every chunk maps to one or two named sections. These are the sections shown in the UI's citations.

---

## 2. Embedding model: `Qwen/Qwen3-Embedding-0.6B` (swap to 4B or 8B with one env var)

| Model | Cost and licence | Why not (for this project) |
|---|---|---|
| **Qwen3-Embedding-0.6B (chosen)** | Free, Apache-2.0, runs on CPU | Smallest model in the strongest open family. 1024-dim, 32K context, instruction-aware queries. |
| Qwen3-Embedding-8B / 4B | Free, Apache-2.0 | Best quality in the family. Qwen3-Embedding-8B scores **69.44** on MTEB English v2 retrieval, above gemini-embedding-001 (64.35) and OpenAI text-embedding-3-large (57.98). Needs a GPU for interactive latency. Use it via `EMBED_MODEL=Qwen/Qwen3-Embedding-8B` when one is available. |
| gemini-embedding-001 | Paid API (about $0.15 per 1M tokens) | Strong and GCP-native, but it is a paid per-token API. Every document and query also leaves your infrastructure. It scores below Qwen3-8B on English retrieval in the same leaderboard. |
| OpenAI text-embedding-3-large | Paid API | Lower retrieval score (57.98) and not free. |
| BGE-M3 | Free, open | Good dense, sparse and multi-vector model, but an older generation that Qwen3 outperforms. |
| EmbeddingGemma-300M | Free, open (Google) | Very small (300M) and built for phones and laptops. A good fallback for very tight memory, with less headroom than Qwen3. |
| all-MiniLM / all-mpnet-base-v2 | Free | Fast and popular, but rank low on the MTEB leaderboard for retrieval. |
| harrier-oss-v1-27b | Free, MIT | Top multilingual retrieval score in October 2026 (78.27), but at 27B parameters it is impractical to serve for a hackathon. |

**Why it fits:** it costs nothing, has no per-query API, and fits in a CPU Cloud Run container. It is instruction-aware: queries get a task instruction like "Given a question about filing police complaints… retrieve passages that answer it", and documents are embedded as-is. The reranker (section 4) recovers most of the precision a bigger embedder would add. Upgrading to the 8B model is a config change, not a code change.

---

## 3. Vector database: Qdrant (embedded on Cloud Run, Qdrant Cloud free tier optional)

| Option | Where it falls short here |
|---|---|
| **Qdrant (chosen)** | Apache-2.0, written in Rust. Filters metadata *during* the ANN search, and has native dense plus sparse hybrid search. **Embedded mode** stores the index on local disk, so we build it at Docker build time and bake it into the image. The result is no database server and **$0**. The same client code switches to **Qdrant Cloud's free 1 GB cluster** (no card) by setting `QDRANT_URL`. |
| FAISS | A library, not a database. No metadata storage, filtering only after search, and no hybrid search, so you would build persistence and payloads yourself. |
| Chroma | Easiest local setup, but aimed at prototypes and small apps, and its hybrid search support is less mature. Qdrant gives the same zero-setup local experience plus a production path. |
| Pinecone | The free Starter plan is generous (2 GB, 5 indexes), but it is managed-only and closed-source. It adds a network hop and a vendor account, and cannot be baked into the container. |
| Vertex AI Vector Search | GCP-native, but needs a deployed index endpoint that is billed while running. That is overkill and not free for 40 chunks. |
| pgvector (Cloud SQL / AlloyDB) | Good if you already run Postgres. Here it would mean paying for an always-on database instance. |

**Hybrid retrieval.** Qdrant returns the top 30 dense hits, BM25 (`rank-bm25`) returns the top 30 keyword hits, and **Reciprocal Rank Fusion** (k=60) merges them. Legal questions mix paraphrase ("police won't take my complaint") with exact identifiers ("173(3)", "D.K. Basu", "CCTNS", "Section 41D"). Dense search handles the first and BM25 handles the second. RRF fuses ranks, not scores, so the two incompatible score scales never need calibrating. On the offline evaluation, hybrid beat both single retrievers on Hit@1 (0.94 vs 0.81).

---

## 4. Reranker: `ms-marco-MiniLM-L6-v2` on CPU, `Qwen/Qwen3-Reranker-0.6B` on GPU

A reranker reads the **query and passage together** (cross-attention). The embedder encodes them separately, so the reranker is much more precise. It reorders the top 20 hybrid candidates, and the best 5 go to Gemini.

Reported in the Qwen3 Embedding paper (arXiv 2506.05176, Table 4). Every reranker reorders the same top-100 candidates from Qwen3-Embedding-0.6B:

| Model | Params | MTEB-R (English) | MMTEB-R (multilingual) | MLDR (long docs) |
|---|---|---|---|---|
| *No reranker (Qwen3-Embedding-0.6B alone)* | | *61.82* | *64.64* | *50.26* |
| jina-reranker-v2-base-multilingual | 0.3B | 58.22 | 63.73 | 39.66 |
| gte-multilingual-reranker-base | 0.3B | 59.51 | 59.44 | 66.33 |
| BGE-reranker-v2-m3 | 0.6B | 57.03 | 58.36 | 59.51 |
| **Qwen3-Reranker-0.6B (chosen)** | 0.6B | **65.80** | **66.36** | **67.28** |
| Qwen3-Reranker-4B | 4B | 69.76 | 72.74 | 69.97 |
| Qwen3-Reranker-8B | 8B | 69.02 | 72.94 | 70.19 |

**Why the 0.6B model:** in this setup it is the only reranker of its size that *improves* on first-stage retrieval on every listed benchmark. BGE-reranker-v2-m3, jina-v2 and gte all score *below* the no-reranker baseline on MTEB-R, so they make the ranking worse. Qwen3-Reranker-0.6B is also Apache-2.0, runs on CPU, and accepts a task instruction. The 4B model is better still (+4 MTEB-R) and is one env var away (`RERANK_MODEL=Qwen/Qwen3-Reranker-4B`) on a GPU.

**Why MiniLM is the CPU default.** Qwen3-Reranker-0.6B is a 0.6B-parameter decoder. On a laptop CPU it took about 490 seconds for one question (20 passages, bf16 weights), and still tens of seconds after the float32 and last-token fixes. `cross-encoder/ms-marco-MiniLM-L6-v2` has about 22M parameters (Apache-2.0). On the sentence-transformers benchmark it scores 74.30 NDCG@10 on TREC DL 19 and 39.01 MRR@10 on MS MARCO dev, and processes about 1,800 docs/s on a V100, versus 960 for the L12 model, which scores almost the same. It is English-only and less accurate than Qwen3 on harder benchmarks, but it reranks 10 passages in well under a second on CPU, and a fast reranker that runs is better than an accurate one that times out. Set `RERANK_BACKEND=qwen` on a GPU. Compare the two on this corpus with `python -m scripts.eval`, which prints a `hybrid+rerank` row for whichever reranker is active.

Paid alternatives: **Cohere Rerank 5** and **Voyage rerank-3** are strong managed APIs, but they are not free and send data out. **ColBERTv2** (late interaction) is a good option when you need token-level matching over a large index, but it means a different index type.

**Relevance gate:** the reranker outputs P(relevant), from 0 to 1. If even the best passage scores below `MIN_RELEVANCE` (0.15), the API answers "I couldn't find this in the provided documents" **without calling Gemini**. That saves cost and stops the LLM from improvising on off-topic questions.

---

## 5. LLM: Gemini

| Use | Model | Why |
|---|---|---|
| **Best quality (cost ignored)** | `gemini-4-argon` | Google's flagship reasoning model: stable, 1M-token context, long outputs. Best for tricky multi-passage legal questions. |
| Balanced | `gemini-3.8-flash` | Google's most capable Flash model, and one of the two models its docs recommend for new projects. |
| **Demo default (lowest cost that works)** | `gemini-3.5-flash-lite` | Described as the fastest and most cost-effective 3.5 model, and the other recommended model for new projects. RAG makes the LLM's job easy: read 5 short passages, answer, cite. A Flash-Lite model handles that well. |

Avoid the 2.5 models (restricted access for new projects) and `-preview` endpoints, which can change or be shut down.

**Grounding.** The system prompt requires context-only answers, a `[n]` citation after every factual sentence, exact quoting of sections and deadlines, and a fixed "not found" sentence. Temperature is 0.2.

**Auth.** Locally, use a `GEMINI_API_KEY` from AI Studio. On Cloud Run the same key is stored in **Secret Manager**. You can also use **Vertex AI** with the service account (`GOOGLE_GENAI_USE_VERTEXAI=true`), which needs no key at all.

---

## 6. Cloud: GCP Cloud Run (two services)

- **Backend** (`rag-backend`, Python): models and index are baked into the image. It runs with 2 vCPU, 8 GiB memory and startup CPU boost. It is **private** (`--no-allow-unauthenticated`), so only the frontend's service account has `roles/run.invoker`.
- **Frontend** (`rag-frontend`, Node.js): public. It fetches a Google-signed ID token from the metadata server to call the backend.
- **Cloud Build** builds both images, pushes them to **Artifact Registry**, and deploys. The Gemini key lives in **Secret Manager**.
- Scale to zero by default (`MIN_INSTANCES=0`), which keeps an idle project essentially free. On demo day, set `MIN_INSTANCES=1` so judges never hit a cold start, where the 0.6B models take a while to load on CPU.

---

## Sources

- Chunking: FloTorch 2026, Chroma and NVIDIA benchmarks as summarised by PremAI, "RAG Chunking Strategies: The 2026 Benchmark Guide"; fast.io, "7 Best Chunking Strategies for RAG" (2026); tianpan.co, "Chunking Strategy Is the Hidden Load-Bearing Decision" (2026); arXiv 2606.00881, "Chunking Methods on RAG: Effectiveness vs Computational Cost"
- Embeddings: Supermemory, "Best free embedding models in 2026" (MTEB snapshot, 16 October 2026); techsy.io, "Best embedding models for RAG"; Modal, "Top embedding models on the MTEB leaderboard"; BentoML, "A Guide to Open-Source Embedding Models"
- Rerankers: sentence-transformers pretrained cross-encoders table (sbert.net/docs/cross_encoder/pretrained_models.html); Zhang et al., "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models", arXiv 2506.05176 (Table 4); FutureAGI, "Best Rerankers for RAG in 2026"
- Vector DBs: dev.to, "Qdrant vs Pinecone vs Chroma: Free Vector Database for RAG"; toolhalla.ai and agentwiki vector DB comparisons (2026)
- Gemini: Google AI for Developers, Gemini API models page (ai.google.dev/gemini-api/docs/models), November 2026
