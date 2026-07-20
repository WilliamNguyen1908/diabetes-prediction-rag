"""Shared sentence-embedding helper.

One cached SentenceTransformer instance, reused by the semantic chunker
(rag/chunk.py), the ingest step (rag/ingest.py), and later the query-time
retriever. Embeddings are L2-normalized so cosine similarity == dot product.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, 384-dim; good default for retrieval


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed(texts, batch_size: int = 64) -> np.ndarray:
    """Embed a list of strings -> (n, dim) float32 array, L2-normalized."""
    if not texts:
        return np.empty((0, get_model().get_sentence_embedding_dimension()), dtype=np.float32)
    vecs = get_model().encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)
