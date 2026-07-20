"""Hybrid retrieval: dense (embeddings) + sparse (BM25), fused with RRF.

Why hybrid: dense vectors capture semantic similarity but under-weight exact
tokens; BM25 nails exact lexical matches — drug names (empagliflozin, SGLT2,
GLP-1), dosages, and comorbidity terms (heart failure, stroke) that the
medication-recommendation flow depends on. Reciprocal Rank Fusion (RRF) combines
the two rank lists without needing to reconcile their incompatible score scales.

Usage:
    r = HybridRetriever()          # loads index + builds BM25 once
    hits = r.search("SGLT2 inhibitor for heart failure", k=5)
"""
import os
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from embed import embed
from store import load_index

# Keep alphanumerics + internal hyphens so "glp-1", "sglt2", "a1c", dosages survive.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")

# Cross-encoder reranker: scores (query, chunk) jointly for fine-grained relevance,
# unlike the bi-encoder which embeds them separately. Override via RERANKER_MODEL
# bge-reranker-base is the stronger option (best MRR in our eval); override via RERANKER_MODEL.
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def rerank_chunks(query: str, chunks, top_k: int):
    """Cross-encoder rerank a candidate pool against `query`; return the top_k reordered,
    each annotated with a 'rerank_score'. Used as a final filter before the LLM."""
    if not chunks:
        return chunks
    scores = get_reranker().predict([(query, c["text"]) for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    out = []
    for i in order[:top_k]:
        c = dict(chunks[i])
        c["rerank_score"] = float(scores[i])
        out.append(c)
    return out


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(rank_lists, rrf_k: int = 60):
    """Reciprocal Rank Fusion. rank_lists: list of ranked index sequences (best first).
    Returns list of (index, fused_score) sorted by score desc."""
    scores = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    def __init__(self):
        self.embeddings, self.chunks = load_index()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, k: int = 5, cand: int = 30, rrf_k: int = 60,
               rerank: bool = False, rerank_pool: int = 20):
        """Two-stage retrieval: hybrid recall -> cross-encoder rerank.

        Stage 1 (recall): dense + BM25 fused with RRF -> top `rerank_pool` candidates.
        Stage 2 (precision): a cross-encoder rescores (query, chunk) pairs and the top
        `k` by that score are returned. `cand` = per-retriever fusion pool size.
        Set rerank=False to return the RRF top-k directly (bi-encoder + BM25 only).
        """
        n = len(self.chunks)
        cand = min(cand, n)

        # Dense: cosine (embeddings are normalized) -> top `cand` indices.
        qv = embed([query])[0]
        dense_sims = self.embeddings @ qv
        dense_ranked = np.argsort(-dense_sims)[:cand].tolist()

        # Sparse: BM25 -> top `cand` indices.
        bm25_scores = self.bm25.get_scores(tokenize(query))
        sparse_ranked = np.argsort(-bm25_scores)[:cand].tolist()

        fused = rrf_fuse([dense_ranked, sparse_ranked], rrf_k=rrf_k)
        pool = fused[: (rerank_pool if rerank else k)]

        dense_rank = {idx: r for r, idx in enumerate(dense_ranked)}
        sparse_rank = {idx: r for r, idx in enumerate(sparse_ranked)}
        results = []
        for idx, score in pool:
            results.append({
                **self.chunks[idx],
                "rrf_score": score,
                "dense_score": float(dense_sims[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "dense_rank": dense_rank.get(idx),      # None if not in dense top-`cand`
                "sparse_rank": sparse_rank.get(idx),    # None if not in BM25 top-`cand`
            })

        if rerank and results:
            scores = get_reranker().predict([(query, r["text"]) for r in results])
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            results.sort(key=lambda r: -r["rerank_score"])

        return results[:k]


if __name__ == "__main__":
    r = HybridRetriever()
    for q in ["empagliflozin for patient with heart failure",
              "medication for type 2 diabetes with chronic kidney disease"]:
        print("\n" + "=" * 80)
        print("QUERY:", q)
        for h in r.search(q, k=5):
            d = f"d#{h['dense_rank']}" if h["dense_rank"] is not None else "d#-"
            s = f"b#{h['sparse_rank']}" if h["sparse_rank"] is not None else "b#-"
            head = (h["heading"] or "(no heading)")[:60]
            print(f"  rrf={h['rrf_score']:.4f} [{d} {s}] {h['source_file']} | {head}")
