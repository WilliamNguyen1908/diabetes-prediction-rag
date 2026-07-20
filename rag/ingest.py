"""One-time: knowledge/md/*.md -> chunks -> embeddings -> knowledge/index/.

Run:  uv run python rag/ingest.py   (after rag/convert.py)

Chunks every Markdown doc with the hybrid semantic chunker, embeds each chunk,
and saves the index (embeddings.npy + chunks.json) via rag/store.py.
"""
from pathlib import Path

import numpy as np

from chunk import chunk_markdown
from embed import embed
from store import save_index

ROOT = Path(__file__).resolve().parent.parent
MD_DIR = ROOT / "knowledge" / "md"


def main() -> None:
    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        print(f"No Markdown found in {MD_DIR}. Run rag/convert.py first.")
        return

    all_chunks = []
    print(f"Chunking {len(md_files)} document(s)...\n")
    for md in md_files:
        text = md.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source_file=md.name)
        all_chunks.extend(chunks)
        print(f"  {md.name:14} -> {len(chunks):>4} chunks")

    if not all_chunks:
        print("No chunks produced — check conversion output.")
        return

    sizes = [len(c["text"]) for c in all_chunks]
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Chunk chars — min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)}")

    print("\nEmbedding chunks...")
    vecs = embed([c["text"] for c in all_chunks])
    print(f"Embeddings: {vecs.shape}")

    save_index(np.asarray(vecs), all_chunks)
    print(f"\nSaved index to {ROOT / 'knowledge' / 'index'} (embeddings.npy + chunks.json)")


if __name__ == "__main__":
    main()
