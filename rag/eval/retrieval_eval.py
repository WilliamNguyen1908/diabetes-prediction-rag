"""Fast retrieval evaluation: hybrid vs dense, using SEMANTIC (embedding) matching.

Why not RAGAS's non-LLM context metrics: they compare retrieved vs reference
contexts by *string distance*. Our chunks carry heading prefixes and different
whitespace than the testset's `reference_contexts` excerpts, so string matching
scores ~0 even for clearly-relevant chunks (verified: a chunk with cosine
similarity 0.77 to the reference still string-matches near 0). And RAGAS's
LLM-judged context metrics need the local judge, which is impractically slow here.

So we score retrieval semantically with the same embedding model used to build the
index — fast, deterministic, no LLM:
  - coverage:  mean over reference_contexts of the max cosine similarity to any
               retrieved chunk (threshold-free "how well retrieval covers refs").
  - recall@k:  fraction of reference_contexts covered above a similarity threshold.
  - hit@k:     fraction of questions with >=1 reference covered above threshold.
  - mrr:       1/rank of the first retrieved chunk matching any reference.

Run:  uv run python rag/eval/retrieval_eval.py [k] [threshold]
Output: rag/eval/retrieval_results.json  (+ printed hybrid-vs-dense table)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
from embed import embed          # noqa: E402
from retrieve import HybridRetriever  # noqa: E402
from store import load_index     # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR / "testset.json"


def score_question(retrieved_texts, reference_contexts, threshold):
    """Semantic retrieval metrics for one question."""
    C = embed(retrieved_texts)                       # (k, d), normalized
    R = embed(reference_contexts)                    # (m, d), normalized
    sims = R @ C.T                                    # (m refs, k retrieved) cosine
    best_per_ref = sims.max(axis=1)                  # best retrieved match per reference
    coverage = float(best_per_ref.mean())
    recall = float((best_per_ref >= threshold).mean())
    hit = float((best_per_ref >= threshold).any())
    # MRR: rank of first retrieved chunk that matches ANY reference above threshold
    matched_ranks = [j for j in range(C.shape[0]) if (sims[:, j] >= threshold).any()]
    mrr = 1.0 / (matched_ranks[0] + 1) if matched_ranks else 0.0
    return {"coverage": coverage, "recall": recall, "hit": hit, "mrr": mrr}


# mode -> how to retrieve k chunk texts for a query
def _retrieve(mode, retriever, embeddings, chunks, q, k):
    if mode == "dense":
        idx = np.argsort(-(embeddings @ embed([q])[0]))[:k]
        return [chunks[i]["text"] for i in idx]
    if mode == "hybrid":
        return [h["text"] for h in retriever.search(q, k=k, rerank=False)]
    if mode == "rerank":  # hybrid recall + cross-encoder rerank
        return [h["text"] for h in retriever.search(q, k=k, rerank=True)]
    raise ValueError(mode)


def run_mode(testset, mode, k, threshold):
    retriever = HybridRetriever() if mode in ("hybrid", "rerank") else None
    embeddings, chunks = load_index() if mode == "dense" else (None, None)
    rows = []
    for item in testset:
        q = item["user_input"]
        retrieved = _retrieve(mode, retriever, embeddings, chunks, q, k)
        m = score_question(retrieved, item["reference_contexts"], threshold)
        m["pillar"] = item.get("metadata", {}).get("pillar")
        rows.append(m)
    return rows


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ("coverage", "recall", "hit", "mrr")}


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    print(f"Semantic retrieval eval | {len(testset)} questions | k={k} | threshold={threshold}\n")
    print(f"  {'mode':7} {'coverage':>9} {'recall@k':>9} {'hit@k':>7} {'mrr':>6}")

    modes = ("dense", "hybrid", "rerank")
    out = {}
    for mode in modes:
        rows = run_mode(testset, mode, k, threshold)
        a = agg(rows)
        out[mode] = {"overall": a, "rows": rows}
        print(f"  {mode:7} {a['coverage']:9.3f} {a['recall']:9.3f} {a['hit']:7.3f} {a['mrr']:6.3f}")

    print("\n  medication-pillar only (n=22):")
    for mode in modes:
        med = [r for r in out[mode]["rows"] if r["pillar"] == "medication"]
        a = agg(med)
        print(f"    {mode:7} {a['coverage']:9.3f} {a['recall']:9.3f} {a['hit']:7.3f} {a['mrr']:6.3f}")

    # Per-pillar breakdown (hybrid mode) — does term specificity predict retrieval quality?
    pillar_mode = "hybrid"
    byp = defaultdict(list)
    for r in out[pillar_mode]["rows"]:
        byp[r["pillar"]].append(r)
    per_pillar = {p: {"n": len(rs), **agg(rs)} for p, rs in byp.items()}
    print(f"\n  per pillar ({pillar_mode} mode), sorted by recall@k:")
    print(f"    {'pillar':13}{'n':>3}{'coverage':>10}{'recall':>9}{'hit':>7}{'mrr':>7}")
    for p in sorted(per_pillar, key=lambda p: -per_pillar[p]["recall"]):
        a = per_pillar[p]
        print(f"    {p:13}{a['n']:>3}{a['coverage']:>10.3f}{a['recall']:>9.3f}{a['hit']:>7.3f}{a['mrr']:>7.3f}")

    (EVAL_DIR / "retrieval_results.json").write_text(
        json.dumps({"k": k, "threshold": threshold,
                    **{m: out[m]["overall"] for m in modes},
                    "per_pillar_hybrid": per_pillar}, indent=2), encoding="utf-8")
    print("\nSaved rag/eval/retrieval_results.json")


if __name__ == "__main__":
    main()
