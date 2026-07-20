"""Tier 1 generator evaluation (no LLM judge):

  1. Answer semantic similarity — cosine(generated answer, reference answer). A cheap
     proxy for correctness. Note: the production generator emits recommendation-style
     prose, so this is a rough alignment signal, not exact-match correctness.
  2. Drug-grounding / hallucination check — every specific medication named in the
     answer must appear in the retrieved context; anything not grounded is flagged.
     This directly guards the medication feature (a hallucinated drug is the scariest
     failure mode for this system).

For each question we retrieve (hybrid) + generate (llama3.1), and SAVE the retrieved
context, the answer, and per-question scores to generator_results.json — so the
retrieved context (otherwise computed live and never persisted) is inspectable.

Run:  uv run python rag/eval/generator_eval.py [k]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
from embed import embed          # noqa: E402
# Reuse the drug-grounding logic from the generator (single source of truth).
from generate import (find_drug_agents, generate_from_chunks,  # noqa: E402
                      ungrounded_agents)
from retrieve import HybridRetriever  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR / "testset.json"

# Real drug dose = number + mg/mcg/units NOT followed by '/' — excludes lab values like
# "126 mg/dL" and "2,300 mg/day" that are not medication doses.
_DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|units?)\b(?!\s*/)", re.I)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    print(f"Generator eval | {len(testset)} questions | k={k}\n")

    records, sims = [], []
    total_drug_mentions = grounded = 0
    dose_flags = 0
    reached_user = 0   # ungrounded drugs that survived the safety filter (should be 0)

    for i, item in enumerate(testset, 1):
        q = item["user_input"]
        chunks = retriever.search(q, k=k)
        context_text = "\n".join(c["text"] for c in chunks).lower()
        out = generate_from_chunks(chunks, stage="the patient's", patient_summary=f"Question: {q}")
        raw = out["raw_recommendations"]     # measure the MODEL, before the safety filter
        answer = out["recommendations"]      # user-facing (filtered)

        # 1. answer similarity to reference (user-facing text)
        sim = float(embed([answer])[0] @ embed([item["reference"]])[0])
        sims.append(sim)

        # 2. drug grounding — measure the raw model output; a true hallucination = neither
        # the agent name nor its class token appears in the retrieved context.
        agents = find_drug_agents(raw)
        ungrounded = ungrounded_agents(raw, context_text)
        total_drug_mentions += len(agents)
        grounded += len(agents) - len(ungrounded)
        reached_user += len(ungrounded_agents(answer, context_text))  # after filter

        has_dose = bool(_DOSE_RE.search(raw))
        dose_flags += int(has_dose)

        records.append({
            "question": q,
            "pillar": item.get("metadata", {}).get("pillar"),
            "reference": item["reference"],
            "answer": answer,
            "retrieved_context": [{"source_file": c["source_file"], "heading": c["heading"], "text": c["text"]} for c in chunks],
            "answer_similarity": sim,
            "drugs_named": agents,
            "drugs_ungrounded": ungrounded,
            "contains_dose": has_dose,
        })
        flag = f"  UNGROUNDED={ungrounded}" if ungrounded else ""
        print(f"  [{i:2}/{len(testset)}] sim={sim:.2f} drugs={agents or '-'}{flag}")

    ground_rate = grounded / total_drug_mentions if total_drug_mentions else 1.0
    summary = {
        "n": len(testset),
        "mean_answer_similarity": float(np.mean(sims)),
        "total_drug_mentions": total_drug_mentions,
        "grounded_mentions": grounded,
        "drug_grounding_rate": ground_rate,
        "ungrounded_reaching_user": reached_user,
        "answers_with_a_dose": dose_flags,
    }
    (EVAL_DIR / "generator_results.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"  mean answer similarity : {summary['mean_answer_similarity']:.3f}")
    print(f"  drug mentions          : {total_drug_mentions} across all answers")
    print(f"  drug grounding (raw)   : {ground_rate:.3f}  ({grounded}/{total_drug_mentions} grounded — agent or class in context)")
    print(f"  ungrounded reaching user: {reached_user}  (after safety filter; want 0)")
    print(f"  answers with a real dose: {dose_flags}  (prompt forbids dosing; want 0)")
    print(f"\nSaved rag/eval/generator_results.json (includes retrieved context per question)")


if __name__ == "__main__":
    main()
