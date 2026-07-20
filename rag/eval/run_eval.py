"""Phases 2-4: run the pipeline over the testset and score it with RAGAS.

For each test question:
  - retrieve contexts (hybrid by default; dense-only with --dense-only)
  - generate an answer with the real generation pipeline (llama3.1)
Then score with RAGAS using the local qwen3-vl:8b judge:
  faithfulness, response relevancy, context precision (w/ reference),
  context recall (w/ reference).

Run:
  uv run python rag/eval/run_eval.py                 # hybrid retrieval
  uv run python rag/eval/run_eval.py --dense-only    # dense-only (comparison)

Output: rag/eval/results_<mode>.json  (+ printed aggregate)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from ragas_local import JUDGE_MODEL, RUN_CONFIG, get_eval_embeddings, get_judge_llm

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
from embed import embed          # noqa: E402
from generate import generate_from_chunks, _retriever  # noqa: E402
from store import load_index     # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR / "testset.json"
K = 6


def dense_chunks(question, embeddings, chunks, k=K):
    qv = embed([question])[0]
    idx = np.argsort(-(embeddings @ qv))[:k]
    return [chunks[i] for i in idx]


def hybrid_chunks(question, k=K):
    return _retriever().search(question, k=k)


def build_samples(testset, dense_only):
    """Retrieve (mode-specific) + generate an answer from THOSE SAME chunks, so the
    dense-vs-hybrid comparison reflects each retrieval set honestly."""
    embeddings, chunks = load_index() if dense_only else (None, None)
    samples = []
    for i, item in enumerate(testset, 1):
        q = item.get("user_input") or item.get("question")
        picked = dense_chunks(q, embeddings, chunks) if dense_only else hybrid_chunks(q)
        answer = generate_from_chunks(picked, stage="the patient's",
                                      patient_summary=f"Question: {q}")["recommendations"]
        print(f"  [{i}/{len(testset)}] answered: {q[:60]}...")
        samples.append(SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=[c["text"] for c in picked],
            reference=item.get("reference"),
        ))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only evaluate the first N questions")
    args = ap.parse_args()
    mode = "dense" if args.dense_only else "hybrid"

    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    if args.limit:
        testset = testset[: args.limit]
    print(f"Loaded {len(testset)} test samples. Mode: {mode}. Generating answers...")
    samples = build_samples(testset, args.dense_only)

    llm, emb = get_judge_llm(), get_eval_embeddings()
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    print(f"Scoring {len(samples)} samples with {JUDGE_MODEL} judge...")
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics,
                      llm=llm, embeddings=emb, run_config=RUN_CONFIG, show_progress=True)

    out = EVAL_DIR / f"results_{mode}.json"
    df = result.to_pandas()
    scores = {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind == "f"}
    out.write_text(json.dumps({"mode": mode, "n": len(samples), "aggregate": scores,
                               "per_sample": df.to_dict(orient="records")},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== {mode} aggregate ===")
    for k, v in scores.items():
        print(f"  {k:32} {v:.3f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
