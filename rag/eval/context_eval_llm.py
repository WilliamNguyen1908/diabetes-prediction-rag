"""LLM-judged retrieval metrics: context precision & recall via Claude (direct calls).

Complements retrieval_eval.py (which uses embedding similarity). Here Claude *reasons*
about relevance:
  - context precision (avg-precision): are the retrieved chunks relevant to the
    question/reference, and are the relevant ones ranked high? (RAGAS-style, with reference)
  - context recall: what fraction of the reference answer's claims are supported by the
    retrieved context?

Direct Claude calls (RAGAS's evaluate() deadlocks with the Anthropic async client).

Run (with ANTHROPIC_API_KEY set):
  uv run python rag/eval/context_eval_llm.py [limit]
Output: rag/eval/context_llm_results.json
"""
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
from retrieve import HybridRetriever      # noqa: E402

from langchain_anthropic import ChatAnthropic  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR / "testset.json"
JUDGE = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
K = 6
_llm = ChatAnthropic(model=JUDGE, temperature=0, max_tokens=1024)

PRECISION_PROMPT = """For the QUESTION and its REFERENCE answer, decide for EACH retrieved CONTEXT passage whether it is relevant/useful for answering the question. Return ONLY a JSON array of booleans, one per context, in order.

QUESTION: {q}
REFERENCE: {ref}

CONTEXTS:
{ctx_list}"""

RECALL_PROMPT = """Break the REFERENCE answer into its distinct factual claims, then decide for each whether it is supported by the retrieved CONTEXT. Return ONLY JSON: {{"supported": <int>, "total": <int>}}.

REFERENCE: {ref}

CONTEXT:
{context}"""


def _parse(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    return json.loads(m.group(0)) if m else None


def average_precision(rels):
    """RAGAS-style: reward relevant chunks ranked higher."""
    rel = [1 if r else 0 for r in rels]
    if sum(rel) == 0:
        return 0.0
    score, hits = 0.0, 0
    for i, r in enumerate(rel):
        if r:
            hits += 1
            score += hits / (i + 1)
    return score / sum(rel)


async def score_question(sem, item, retriever):
    q, ref = item["user_input"], item["reference"]
    chunks = retriever.search(q, k=K)
    ctx_list = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    context = "\n".join(c["text"] for c in chunks)
    async with sem:
        p_resp, r_resp = await asyncio.gather(
            _llm.ainvoke(PRECISION_PROMPT.format(q=q, ref=ref, ctx_list=ctx_list)),
            _llm.ainvoke(RECALL_PROMPT.format(ref=ref, context=context)),
        )
    try:
        rels = _parse(p_resp.content)[:K]
        precision = average_precision(rels)
        simple_precision = sum(bool(r) for r in rels) / len(rels)
    except Exception:
        precision = simple_precision = None
    try:
        rj = _parse(r_resp.content)
        recall = rj["supported"] / rj["total"] if rj.get("total") else 1.0
    except Exception:
        recall = None
    return {"question": q, "pillar": item.get("metadata", {}).get("pillar"),
            "precision": precision, "simple_precision": simple_precision, "recall": recall}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


async def main():
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        testset = testset[: int(sys.argv[1])]
    retriever = HybridRetriever()
    print(f"LLM-judged retrieval eval | {len(testset)} questions | judge {JUDGE}")

    sem = asyncio.Semaphore(4)
    rows = await asyncio.gather(*[score_question(sem, it, retriever) for it in testset])

    byp = defaultdict(list)
    for r in rows:
        byp[r["pillar"]].append(r)
    summary = {
        "judge": JUDGE, "n": len(rows),
        "context_precision": _mean([r["precision"] for r in rows]),
        "simple_precision": _mean([r["simple_precision"] for r in rows]),
        "context_recall": _mean([r["recall"] for r in rows]),
        "per_pillar": {p: {"precision": _mean([r["precision"] for r in rs]),
                           "recall": _mean([r["recall"] for r in rs]), "n": len(rs)}
                       for p, rs in byp.items()},
    }
    (EVAL_DIR / "context_llm_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8")

    print("\n=== LLM-judged retrieval (Claude) ===")
    print(f"  context precision (avg-precision): {summary['context_precision']:.3f}")
    print(f"  simple precision@{K}:               {summary['simple_precision']:.3f}")
    print(f"  context recall:                    {summary['context_recall']:.3f}")
    print("\nSaved rag/eval/context_llm_results.json")


if __name__ == "__main__":
    asyncio.run(main())
