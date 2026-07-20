"""Vector store: persist chunk embeddings + metadata, and cosine top-k search.

Deliberately NOT FAISS. This corpus is tiny (~13 PDFs -> a few thousand chunks),
so brute-force cosine similarity over a NumPy matrix is instant and avoids
faiss-cpu's uncertain Python 3.14 wheels. Embeddings are L2-normalized, so
cosine similarity is a single matrix-vector dot product. Swap in FAISS here later
if the corpus grows by orders of magnitude — callers only use save/load/search.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "knowledge" / "index"
EMB_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def save_index(embeddings: np.ndarray, chunks: list) -> None:
    if len(embeddings) != len(chunks):
        raise ValueError(f"embeddings ({len(embeddings)}) and chunks ({len(chunks)}) length mismatch")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, embeddings.astype(np.float32))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def load_index():
    if not EMB_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Index not found in {INDEX_DIR}. Build it first: uv run python rag/ingest.py"
        )
    embeddings = np.load(EMB_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return embeddings, chunks


def search(query_vec: np.ndarray, embeddings: np.ndarray, chunks: list, k: int = 5):
    """Return the top-k chunks by cosine similarity, each with a 'score' field."""
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    sims = embeddings @ q                          # both sides normalized -> cosine
    k = min(k, len(chunks))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [{**chunks[i], "score": float(sims[i])} for i in top]
