"""Phase 1: build a synthetic RAGAS testset from the guideline Markdown.

RAGAS's TestsetGenerator builds a knowledge graph over the docs and generates
(question, reference answer, reference_contexts) triples using the local judge
LLM + local embeddings. Kept small (default 24) because 8B local generation is slow.

Run:  uv run python rag/eval/gen_testset.py [n]
Output: rag/eval/testset.json
"""
import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from ragas.testset import TestsetGenerator

from ragas_local import get_eval_embeddings, get_judge_llm

ROOT = Path(__file__).resolve().parent.parent.parent
MD_DIR = ROOT / "knowledge" / "md"
OUT = Path(__file__).resolve().parent / "testset.json"


def load_documents():
    docs = []
    for md in sorted(MD_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": md.name}))
    return docs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    docs = load_documents()
    print(f"Loaded {len(docs)} documents; generating {n} test samples (local, slow)...")

    generator = TestsetGenerator(llm=get_judge_llm(), embedding_model=get_eval_embeddings())
    testset = generator.generate_with_langchain_docs(docs, testset_size=n)

    df = testset.to_pandas()
    records = []
    for _, row in df.iterrows():
        records.append({
            "question": row.get("user_input"),
            "reference": row.get("reference"),
            "reference_contexts": list(row.get("reference_contexts") or []),
        })
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} samples to {OUT}")


if __name__ == "__main__":
    main()
