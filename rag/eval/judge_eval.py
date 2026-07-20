"""Tier 2: RAGAS LLM-judged generation metrics, reusing already-generated answers.

Loads generator_results.json (question, user-facing answer, retrieved context,
reference) produced by generator_eval.py, so we do NOT regenerate with llama3.1 —
we only score. The judge is configured in ragas_local.py (default Claude Sonnet 4.6);
embeddings stay local.

Metrics:
  - faithfulness            : are the answer's claims entailed by the retrieved context?
  - answer relevancy        : does the answer address the question?
  - context precision (ref) : are the retrieved chunks the relevant ones?
  - context recall          : did retrieval cover what the reference needs?

Run:  ANTHROPIC_API_KEY=... uv run python rag/eval/judge_eval.py [limit]
Output: rag/eval/judge_results.json
"""
import json
import sys
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                           LLMContextRecall, ResponseRelevancy)

from ragas_local import JUDGE_MODEL, RUN_CONFIG, get_eval_embeddings, get_judge_llm

EVAL_DIR = Path(__file__).resolve().parent
GEN_RESULTS = EVAL_DIR / "generator_results.json"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    data = json.loads(GEN_RESULTS.read_text(encoding="utf-8"))["records"]
    if limit:
        data = data[:limit]

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],                                   # user-facing (filtered) answer
            retrieved_contexts=[c["text"] for c in r["retrieved_context"]],
            reference=r["reference"],
        )
        for r in data
    ]
    print(f"Judging {len(samples)} pre-generated answers with {JUDGE_MODEL} ...")

    llm, emb = get_judge_llm(), get_eval_embeddings()
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics,
                      llm=llm, embeddings=emb, run_config=RUN_CONFIG, show_progress=True)

    df = result.to_pandas()
    scores = {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind == "f"}
    (EVAL_DIR / "judge_results.json").write_text(
        json.dumps({"judge": JUDGE_MODEL, "n": len(samples), "aggregate": scores,
                    "per_sample": df.to_dict(orient="records")},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n=== RAGAS generation metrics (judge: {JUDGE_MODEL}) ===")
    for k, v in scores.items():
        print(f"  {k:38} {v:.3f}")
    print(f"\nSaved rag/eval/judge_results.json")


if __name__ == "__main__":
    main()
