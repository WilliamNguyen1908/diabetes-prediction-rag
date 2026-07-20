"""A/B the generator: llama3.1 (local) vs Claude Sonnet 4.6, judged identically.

For each generator model, generate an answer for every testset question with the CURRENT
pipeline (same retrieval, prompt, safety filter), then score all answers with the same
Claude faithfulness/relevancy judge (reused from judge_direct). Writes a side-by-side
comparison so the numbers are apples-to-apples.

Run (with ANTHROPIC_API_KEY set):
  GROUNDING_NLI=0 uv run python rag/eval/compare_generators.py
Output: rag/eval/generator_comparison.json
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
import generate as G                       # noqa: E402  (mutated to switch models)
from retrieve import HybridRetriever       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_direct import _score_one        # noqa: E402  (Claude faithfulness + relevancy)

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR / "testset.json"
OUT = EVAL_DIR / "generator_comparison.json"
MODELS = ["llama3.1", "claude-sonnet-4-6"]
K = 6


def generate_answers(model, testset, retriever):
    """Switch the generator model at runtime and produce an answer per question."""
    G.MODEL = model
    G._IS_CLAUDE = model.startswith("claude")
    recs = []
    for i, item in enumerate(testset, 1):
        q = item["user_input"]
        chunks = retriever.search(q, k=K)
        out = G.generate_from_chunks(chunks, stage="the patient's", patient_summary=f"Question: {q}")
        recs.append({
            "question": q,
            "pillar": item.get("metadata", {}).get("pillar"),
            "answer": out["recommendations"],
            "retrieved_context": [{"text": c["text"]} for c in chunks],
            "reference": item["reference"],
        })
        print(f"  [{model}] generated {i}/{len(testset)}", flush=True)
    return recs


async def judge_all(recs):
    sem = asyncio.Semaphore(4)
    return await asyncio.gather(*[_score_one(sem, r) for r in recs])


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


async def main():
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        testset = testset[: int(sys.argv[1])]
    retriever = HybridRetriever()
    results = {}

    for model in MODELS:
        print(f"=== GENERATING with {model} ({len(testset)} questions) ===", flush=True)
        recs = generate_answers(model, testset, retriever)
        print(f"=== JUDGING {model} with Claude ===", flush=True)
        scored = await judge_all(recs)

        byp = defaultdict(lambda: {"f": [], "r": []})
        for s in scored:
            if s["faithfulness"] is not None:
                byp[s["pillar"]]["f"].append(s["faithfulness"])
            if s["answer_relevancy"] is not None:
                byp[s["pillar"]]["r"].append(s["answer_relevancy"])

        results[model] = {
            "faithfulness": _mean([s["faithfulness"] for s in scored]),
            "answer_relevancy": _mean([s["answer_relevancy"] for s in scored]),
            "per_pillar": {p: {"faithfulness": _mean(v["f"]), "answer_relevancy": _mean(v["r"]),
                               "n": len(v["f"])} for p, v in byp.items()},
        }
        OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")  # incremental save
        print(f"--- {model}: faithfulness={results[model]['faithfulness']:.3f} "
              f"relevancy={results[model]['answer_relevancy']:.3f}", flush=True)

    print("\n=== COMPARISON ===")
    print(f"  {'model':22}{'faithfulness':>14}{'relevancy':>12}")
    for m in MODELS:
        print(f"  {m:22}{results[m]['faithfulness']:>14.3f}{results[m]['answer_relevancy']:>12.3f}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
