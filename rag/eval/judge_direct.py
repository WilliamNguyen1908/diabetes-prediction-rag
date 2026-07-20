"""Tier 2 generation metrics via DIRECT Claude calls (bypassing RAGAS's executor).

RAGAS's evaluate()/metric internals deadlock on async Anthropic calls with our pinned
langchain 0.3.x stack (a single metric hangs ~60s/call though a direct Claude call is
~1s). Rather than fight that, we implement the two standard metrics directly — the
definitions are simple and this is fast + reliable:

  - faithfulness    : fraction of the answer's factual claims that are supported by the
                      retrieved context (Claude extracts claims + judges support).
  - answer_relevancy: how well the answer addresses the question (0-1, Claude rates).

Judges pre-generated answers from generator_results.json (no llama3.1 regeneration).
Embeddings/generation stay local; only judging goes to Claude. No patient PII in the
eval data.

Run:  (with ANTHROPIC_API_KEY set)  uv run python rag/eval/judge_direct.py [limit]
Output: rag/eval/judge_results.json
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from langchain_anthropic import ChatAnthropic

EVAL_DIR = Path(__file__).resolve().parent
GEN_RESULTS = EVAL_DIR / "generator_results.json"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
CONCURRENCY = 4

_llm = ChatAnthropic(model=JUDGE_MODEL, temperature=0, max_tokens=1500)

FAITHFULNESS_PROMPT = """You evaluate whether an ANSWER is grounded in the provided CONTEXT.
1. Extract the distinct factual/clinical claims asserted in the ANSWER (ignore generic advice that makes no factual assertion, and ignore the safety disclaimer).
2. For each claim, decide if it is supported by the CONTEXT.
Return ONLY JSON: {{"total": <int>, "supported": <int>}}

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEVANCY_PROMPT = """Rate from 0.0 to 1.0 how well the ANSWER addresses the QUESTION (1.0 = directly and completely answers it; 0.0 = unrelated). A thorough answer that includes the correct information should score high even if it adds extra detail.
Return ONLY JSON: {{"score": <float>}}

QUESTION: {question}

ANSWER:
{answer}"""


def _parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


async def _score_one(sem, rec):
    context = "\n".join(c["text"] for c in rec["retrieved_context"])
    async with sem:
        f_resp, r_resp = await asyncio.gather(
            _llm.ainvoke(FAITHFULNESS_PROMPT.format(context=context, answer=rec["answer"])),
            _llm.ainvoke(RELEVANCY_PROMPT.format(question=rec["question"], answer=rec["answer"])),
        )
    try:
        fj = _parse_json(f_resp.content)
        faith = fj["supported"] / fj["total"] if fj.get("total") else 1.0
    except Exception:
        faith = None
    try:
        relevancy = float(_parse_json(r_resp.content)["score"])
    except Exception:
        relevancy = None
    return {"question": rec["question"], "pillar": rec.get("pillar"),
            "faithfulness": faith, "answer_relevancy": relevancy}


async def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    data = json.loads(GEN_RESULTS.read_text(encoding="utf-8"))["records"]
    if limit:
        data = data[:limit]
    print(f"Judging {len(data)} answers with {JUDGE_MODEL} (direct calls)...")

    sem = asyncio.Semaphore(CONCURRENCY)
    rows = await asyncio.gather(*[_score_one(sem, r) for r in data])

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {"judge": JUDGE_MODEL, "n": len(rows),
               "faithfulness": mean("faithfulness"),
               "answer_relevancy": mean("answer_relevancy"),
               "n_scored_faithfulness": sum(r["faithfulness"] is not None for r in rows),
               "n_scored_relevancy": sum(r["answer_relevancy"] is not None for r in rows)}
    (EVAL_DIR / "judge_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Generation metrics (judge: {JUDGE_MODEL}) ===")
    print(f"  faithfulness    : {summary['faithfulness']:.3f}  (n={summary['n_scored_faithfulness']})")
    print(f"  answer_relevancy: {summary['answer_relevancy']:.3f}  (n={summary['n_scored_relevancy']})")
    print("\nSaved rag/eval/judge_results.json")


if __name__ == "__main__":
    asyncio.run(main())
