# Diabetes Prediction + RAG recommendations — runtime image.
#
# Prerequisite: the vector index must exist locally before building
# (knowledge/index/{embeddings.npy,chunks.json} — build it once with
#  `uv run python rag/convert.py && uv run python rag/ingest.py`).
# The image copies that prebuilt index; PDFs and the training CSV are not shipped.
#
# Build:  docker build -t diabetes-rag .
# Run:    docker compose up   (starts this app + an Ollama service; see docker-compose.yml)
#
# Generation needs an Ollama server. Point the app at it with OLLAMA_HOST
# (compose sets http://ollama:11434). To use Claude instead:
#   GENERATOR_MODEL=claude-sonnet-4-6 + ANTHROPIC_API_KEY.

# Python 3.12, not 3.14: the model-compat pins (pandas 2.2.2, scikit-learn 1.6.1)
# have no cp314 wheels — on 3.14 they'd need a full compiler toolchain in the
# image. On 3.12 everything installs from prebuilt wheels. (Local dev stays on
# 3.14, where uv builds those two from source.)
FROM python:3.12-slim

# libgomp: OpenMP runtime XGBoost needs on Linux (the brew-libomp equivalent)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch FIRST, from the PyTorch CPU index — the default PyPI resolution
# on Linux drags in ~5 GB of CUDA wheels this app never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Runtime dependencies (mirrors pyproject [project.dependencies], minus
# notebook/viz and pymupdf4llm — those are build/exploration tools, not runtime).
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    scikit-learn==1.6.1 \
    xgboost==3.3.0 \
    numpy \
    joblib \
    "fastapi>=0.139.0" \
    "uvicorn[standard]>=0.50.0" \
    "jinja2>=3.1.6" \
    "python-multipart>=0.0.32" \
    "sentence-transformers>=5.6.0" \
    "pysbd>=0.3.4" \
    "rank-bm25>=0.2.2" \
    "ollama>=0.6.2" \
    "anthropic>=0.116.0" \
    sentencepiece

# Pre-download the three Hugging Face models (embedder, reranker, NLI) into the
# image so the first request doesn't spend minutes downloading ~2 GB.
# Skip with:  docker build --build-arg PRELOAD_MODELS=0 .
ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('BAAI/bge-reranker-base'); \
CrossEncoder('MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')"; \
    fi

WORKDIR /app

# App code + model artifacts + prebuilt vector index
COPY app.py predict.py db.py ./
COPY diabetes_xgb.json preprocess_bundle.joblib ./
COPY templates/ templates/
COPY rag/*.py rag/
COPY knowledge/index/ knowledge/index/

# Audit log goes to a mounted volume (see compose); db.py honors this override.
ENV RECOMMEND_DB=/data/recommendation_log.db
RUN mkdir -p /data

EXPOSE 8000
# Honor $PORT if the platform injects one (Cloud Run sets it, default 8080),
# otherwise fall back to 8000 (compose / HF Space / local). `exec` makes uvicorn
# PID 1 so it receives SIGTERM for graceful shutdown.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
