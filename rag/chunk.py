"""Hybrid semantic chunker for the converted Markdown guideline docs.

Fixed-size windows are avoided on purpose: for clinical guidelines they cut
mid-recommendation and separate a threshold/dose from its condition. Instead:

  1. Structure-aware split — break each doc on Markdown headings (which
     pymupdf4llm recovers). Each heading section is a coherent unit, and the
     heading trail (H1 > H2 > H3) is kept as context on every chunk.
  2. Embedding-based semantic split — within a section that exceeds the size
     guardrail, split at *semantic breakpoints*: embed each segment (sentence
     or table/paragraph), and start a new chunk where the cosine distance to the
     previous segment jumps past a percentile threshold, or the size cap is hit.
  3. Guardrails — hard char cap so no chunk overflows the embedder window; tiny
     trailing chunks merge back into the previous one; tables are never
     sentence-split.

Public API: chunk_markdown(text, source_file) -> list[dict(text, source_file, heading)].
"""
import re

import numpy as np
import pysbd

from embed import embed

MAX_CHARS = 1200      # hard cap per chunk
MIN_CHARS = 250       # merge chunks smaller than this into the previous one
TARGET_CHARS = 900    # soft target that triggers a semantic-break check
BREAK_PCTL = 85       # semantic-distance percentile above which to break

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SEG = pysbd.Segmenter(language="en", clean=False)

# Non-content sections to drop entirely (citation lists, front/back matter) — they
# pollute retrieval with reference strings instead of clinical guidance.
_SKIP_HEADINGS = {
    "references", "bibliography", "acknowledgments", "acknowledgements",
    "funding", "author contributions", "conflicts of interest",
    "duality of interest", "disclosures",
}


def _is_skippable(heading_trail: str) -> bool:
    last = heading_trail.split(" > ")[-1].strip().lower() if heading_trail else ""
    return last in _SKIP_HEADINGS

# Conservative noise filters — running headers/footers and extraction artifacts.
_NOISE_RES = [
    re.compile(r"^diabetesjournals\.org", re.I),
    re.compile(r"^downloaded from", re.I),
    re.compile(r"^https?://"),
    re.compile(r"^<!--.*-->$"),
    re.compile(r"^S\d+\b.*Diabetes", re.I),          # "S28 Diagnosis and Classification of Diabetes"
    re.compile(r".*Diabetes\s+S\d+\s*$", re.I),      # "Diagnosis and Classification of Diabetes S29"
    re.compile(r"^S\d+\s*$"),                        # lone page marker "S27"
]


def _clean_heading(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text)           # strip markdown emphasis
    return re.sub(r"\s+", " ", text).strip()


def _is_noise(line: str) -> bool:
    s = line.strip()
    return any(r.search(s) for r in _NOISE_RES)


def _iter_sections(text: str):
    """Yield (heading_trail, body_text) using a heading-level stack for context."""
    stack = []  # list of (level, clean_text)
    buf = []

    def trail():
        return " > ".join(t for _, t in stack)

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            if buf:
                yield trail(), "\n".join(buf)
                buf = []
            level = len(m.group(1))
            clean = _clean_heading(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            if clean:
                stack.append((level, clean))
        elif not _is_noise(line):
            buf.append(line)
    if buf:
        yield trail(), "\n".join(buf)


def _segments(body: str): # this code return tables in the md file
    """Split a section body into atomic segments: sentences, but tables/list rows kept whole."""
    segs = []
    for para in re.split(r"\n\s*\n", body):      # paragraphs on blank lines
        para = para.strip()
        if not para:
            continue
        if "|" in para or para.startswith(("-", "*", ">")):
            segs.append(para)                    # table / list / quote: keep intact
        else:
            segs.extend(s.strip() for s in _SEG.segment(para) if s.strip())
    return segs 


def _hard_wrap(seg: str):
    """Split an oversized single segment (e.g. a big table) on char boundaries."""
    return [seg[i:i + MAX_CHARS] for i in range(0, len(seg), MAX_CHARS)]


def _semantic_groups(segs):
    """Group segments into chunks at size/semantic breakpoints. Returns list[str]."""
    # Expand any segment already larger than the cap.
    expanded = []
    for s in segs:
        expanded.extend(_hard_wrap(s) if len(s) > MAX_CHARS else [s])
    segs = expanded
    if len(segs) <= 1:
        return ["\n".join(segs)] if segs else []

    vecs = embed(segs)                            # normalized -> cosine = dot
    dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)   # distance between consecutive segs
    threshold = float(np.percentile(dists, BREAK_PCTL)) if len(dists) else 1.0

    groups, cur, cur_len = [], [], 0
    for i, seg in enumerate(segs):
        add = len(seg) + (1 if cur else 0)
        hard = cur_len + add > MAX_CHARS
        semantic = i > 0 and cur_len >= TARGET_CHARS and dists[i - 1] >= threshold
        if cur and (hard or semantic):
            groups.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += add
    if cur:
        groups.append("\n".join(cur))

    # Merge undersized trailing chunk into the previous one.
    if len(groups) >= 2 and len(groups[-1]) < MIN_CHARS:
        last = groups.pop()
        groups[-1] = groups[-1] + "\n" + last
    return groups


def chunk_markdown(text: str, source_file: str):
    """Convert one Markdown document into a list of retrieval chunks."""
    chunks = []
    for heading, body in _iter_sections(text):
        body = body.strip()
        if not body or _is_skippable(heading):
            continue
        if len(body) <= MAX_CHARS:
            groups = [body]
        else:
            groups = _semantic_groups(_segments(body))
        for g in groups:
            g = g.strip()
            if len(g) < 40:                      # drop trivial fragments
                continue
            prefixed = f"{heading}\n\n{g}" if heading else g
            chunks.append({"text": prefixed, "source_file": source_file, "heading": heading})
    return chunks
