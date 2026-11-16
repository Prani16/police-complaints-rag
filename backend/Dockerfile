# Backend image: FastAPI + Qwen3 embedding + MiniLM reranker (CPU) + Qdrant (embedded).
# For the Qwen3 reranker: --build-arg RERANK_BACKEND=qwen --build-arg RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
# Build context is the REPO ROOT (needs data/raw):
#   docker build -f backend/Dockerfile -t rag-backend .
#
# Models are downloaded and the index is built at *build* time, so a cold
# start on Cloud Run only loads weights from local disk - no Hugging Face
# download and no re-embedding on every new instance.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

ARG EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
ARG RERANK_BACKEND=fast
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
ENV EMBED_MODEL=${EMBED_MODEL} RERANK_MODEL=${RERANK_MODEL} RERANK_BACKEND=${RERANK_BACKEND}
RUN python -c "from huggingface_hub import snapshot_download as d; d('${EMBED_MODEL}'); d('${RERANK_MODEL}')"

COPY backend /app/backend
COPY data/raw /app/data/raw
WORKDIR /app/backend

# Part 1 + Part 2 at build time -> index baked into the image.
RUN EMBED_BACKEND=qwen python -m scripts.ingest && EMBED_BACKEND=qwen python -m scripts.build_index

ENV HF_HUB_OFFLINE=1 EMBED_BACKEND=qwen PORT=8080
EXPOSE 8080
# 1 worker: the embedded Qdrant store holds a file lock, and the models are
# loaded once per process. Scale with Cloud Run instances, not workers.
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1
