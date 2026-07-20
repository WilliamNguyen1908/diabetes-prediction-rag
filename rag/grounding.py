"""Runtime NLI grounding check (#4).

Offline faithfulness (rag/eval) *measures* quality on a testset; this *gates every
live response*: it splits the generated answer into claims and, for each, checks whether
the retrieved context entails it (Natural Language Inference). Ungrounded claims are
surfaced (and can be redacted) before the text reaches the patient.

Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli — a small, local NLI cross-encoder
(premise = context chunk, hypothesis = claim -> entailment/neutral/contradiction).

Efficiency: NLI is O(claims x premises), so we don't test every claim against every
chunk. We embed-prefilter to each claim's most-similar chunks (cheap) and run NLI only
on those. A claim is grounded if ANY of its top premises entails it.
"""
from functools import lru_cache

import numpy as np
import pysbd

from embed import embed

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
TOP_PREMISES = 4          # NLI each claim only against its most-similar chunks
# Low threshold on purpose: per-claim NLI is noisy on personalized medical prose
# (compound "your BMI is 33 + guideline advice" sentences score mid-range), so this is a
# TRIPWIRE for clearly-unentailed claims (blatant fabrications score ~0), used as an
# advisory flag — not an auto-redact gate.
ENTAIL_THRESHOLD = 0.15
_SEG = pysbd.Segmenter(language="en", clean=False)

# Preamble / meta sentences that aren't factual claims — skip them.
_SKIP_PREFIXES = ("based on the provided", "here are", "here is", "i will provide",
                  "the following are", "in summary", "note:")


@lru_cache(maxsize=1)
def _nli():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(NLI_MODEL)


@lru_cache(maxsize=1)
def _entail_idx():
    id2label = _nli().model.config.id2label
    for i, label in id2label.items():
        if "entail" in str(label).lower():
            return int(i)
    return 0


def _claims(text: str):
    """Substantive claim sentences from the answer (skip headings, bullets markers,
    the disclaimer, and trivially short lines)."""
    claims = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">") or s.startswith("|"):
            continue
        s = s.lstrip("-*•> ").strip()
        for sent in _SEG.segment(s):
            sent = sent.strip()
            low = sent.lower()
            if len(sent.split()) < 5:
                continue
            if "not a substitute for professional" in low:            # the disclaimer
                continue
            if "do not provide specific medication guidance" in low:  # our own no-med meta line
                continue
            if low.startswith(_SKIP_PREFIXES):                        # preamble / meta
                continue
            claims.append(sent)
    return claims


def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[None, :]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def check_grounding(text: str, chunks, threshold: float = ENTAIL_THRESHOLD, extra_premises=None):
    """Return {'claims': [{claim, entailment, grounded}], 'ungrounded': [claim, ...]}.

    `extra_premises` (e.g. the PATIENT PROFILE) are added to the guideline context as
    legitimate grounding sources, so personalized sentences that cite the patient's own
    values ('your BMI of 33') aren't false-flagged for being absent from the guidelines.
    """
    claims = _claims(text)
    if not claims or not chunks:
        return {"claims": [], "ungrounded": []}

    ctx_texts = [c["text"] for c in chunks] + [p for p in (extra_premises or []) if p]
    ctx_vecs = embed(ctx_texts)                 # normalized
    claim_vecs = embed(claims)
    ce, ei = _nli(), _entail_idx()

    results, ungrounded = [], []
    for claim, cv in zip(claims, claim_vecs):
        sims = ctx_vecs @ cv
        top = np.argsort(-sims)[:TOP_PREMISES]
        probs = _softmax(ce.predict([(ctx_texts[i], claim) for i in top]))
        max_ent = float(probs[:, ei].max())
        grounded = max_ent >= threshold
        results.append({"claim": claim, "entailment": max_ent, "grounded": grounded})
        if not grounded:
            ungrounded.append(claim)
    return {"claims": results, "ungrounded": ungrounded}
