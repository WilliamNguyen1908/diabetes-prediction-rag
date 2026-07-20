# Project Experience Log

A narrative of what we built, the decisions we made, and *why* — so the reasoning
behind the current state isn't lost. Newest work is at the bottom.

## Goal

Predict a patient's diabetes stage from clinical/lifestyle inputs with an ML model,
expose it through a web UI, and after the prediction use RAG over clinical guidelines
to generate personalized diet/lifestyle + medication recommendations — all runnable
locally.

---

## 1. Environment setup (the Python-version saga)

**What happened:** matplotlib/seaborn "wouldn't load" in the notebook. Root cause: the
machine has **three Pythons** (3.9, 3.11, 3.14). The notebook kernel ran on 3.14, but
terminal `pip3`/`python3` pointed at 3.11, so installs never reached the kernel. Also,
what looked like an installed `matplotlib` was actually `matplotlib-inline` (a tiny
helper bundled with ipykernel).

**Decisions & why:**
- **Standardize on Python 3.14** via a single `uv`-managed `.venv`, and point the
  notebook/IDE at it. One interpreter for terminal + notebook + app kills the whole
  class of "installed but can't import" bugs. `.python-version` pins `3.14`.
- **Use `uv`** as the package manager (a `.venv` + `uv.lock` already existed).
- Install into the notebook's interpreter with `!{sys.executable} -m pip install` to
  avoid future kernel/terminal mismatches.

**Gotcha:** `uv pip sync` treats requirements.txt as a *fully resolved lockfile* and
stripped pandas's transitive deps (pytz, dateutil). Use `uv pip install -r` for a
hand-written requirements file; `sync` only for a compiled lock.

---

## 2. Dependency management & the trained model

**What happened:** the model was trained on **Google Colab** (XGBoost). A saved model
is tied to the library versions that created it.

**Decisions & why:**
- **Pin the model-critical libs to the Colab versions** (`pandas==2.2.2`,
  `scikit-learn==1.6.1`, `xgboost==3.3.0`) so the saved artifact loads without
  `InconsistentVersionWarning`/breakage. Other libs left unpinned.
- **`pyproject.toml` is the single source of truth** (uv reads it); requirements.txt
  was a bloated copy-paste (geopandas, google-cloud, duplicate `pandas`) that had
  silently downgraded pandas — cleaned out.
- Bumped `requires-python` to `>=3.12` because xgboost 3.3.0 dropped 3.11, and uv
  resolves against the *whole* declared range, not just the active interpreter.
- **macOS gotcha:** xgboost needs the OpenMP runtime — `brew install libomp`.

**Model artifacts:** `diabetes_xgb.json` (XGBoost native format — version-portable,
better than pickle) + `preprocess_bundle.joblib` (scaler, feature order, one-hot
categories, defaults). `predict.py` reproduces the notebook's exact preprocessing so
training and serving transform inputs identically.

---

## 3. Prediction app (FastAPI + UI)

**Decisions & why:**
- **FastAPI + Jinja2 server-rendered HTML**, Pydantic-validated inputs (important for
  medical fields). Two endpoints, not one: `/predict` (fast, deterministic) separate
  from `/recommend` (slow, non-deterministic LLM) — cleaner to test/cache and the UI
  can show the prediction instantly, then load recommendations.
- Form is **generated from the model's own bundle metadata** (19 numeric + 3 binary +
  6 categorical = 28 fields) so it can never drift from what the model expects.
- **Bug found & fixed:** installed Starlette 1.3.1 requires the new
  `TemplateResponse(request, name, context)` signature; the old order passed the
  context dict as the template name → an unhashable-key 500.

---

## 4. RAG ingestion (PDF → chunks → index)

**markitdown → pymupdf4llm (converter switch).** The user chose markitdown, but it
extracts these **multi-column ADA journal PDFs** in jumbled reading order (interleaved
columns, no headings) — poison for retrieval. `pymupdf4llm` is layout-aware: recovers
reading order, paragraphs, headings, and tables (52 headings vs 0 on the same doc). It
also dropped markitdown's messy `magika`/`onnxruntime` override needed to install on
3.14. **Quality of extraction mattered more than the specific tool.**

**Semantic chunking (not fixed-size).** For clinical guidelines, fixed windows cut
mid-recommendation and separate a dose/threshold from its condition. `chunk.py` does a
**hybrid**: split on Markdown headings (keeping the `H1 > H2 > H3` trail as context on
every chunk), and within oversized sections split at **sentence-embedding semantic
breakpoints** (percentile of consecutive cosine distance) with a hard size cap. Tables
kept intact; `References`/front-back-matter dropped (they polluted retrieval with
citation strings). Result: ~1,300 chunks.

**NumPy cosine store, not FAISS.** The corpus is tiny; a matrix dot product is instant
and avoids faiss-cpu's missing Python 3.14 wheels. Swap point isolated in `store.py`.

**Embeddings:** `sentence-transformers all-MiniLM-L6-v2` (small, fast, 384-dim,
normalized). Resolved fine on 3.14 (torch 2.12) — the Ollama-embeddings fallback wasn't
needed.

---

## 5. Hybrid retrieval (dense + BM25 + RRF)

**Why:** dense vectors capture semantics but under-weight *exact tokens* — drug names
(`empagliflozin`, `SGLT2`, `GLP-1`), dosages, comorbidity terms (`heart failure`,
`stroke`) — which the medication logic depends on. BM25 nails those. **Reciprocal Rank
Fusion** merges the two rank lists without reconciling their incompatible score scales
(`1/(rrf_k + rank)`, `rrf_k=60`). Tokenizer keeps internal hyphens/digits so
`glp-1`/`a1c`/`sglt2` survive.

**Validated:** on a `tirzepatide` query, RRF surfaced chunks dense ranked #7 and #13 (via
BM25 #0/#2) into the top-5 — the concrete win. `retrieve.py::HybridRetriever` builds
BM25 once at construction.

---

## 6. Generation (llama3.1 via Ollama)

**Decisions & why:**
- **Local llama3.1** for generation (no patient data leaves the machine); a small model
  is enough for this corpus size.
- **Dual-query retrieval** (a lifestyle query + a medication query, with comorbidities
  folded into the latter) so both diet and drug context are retrieved and merged.
- **Grounding-first prompt:** use ONLY provided context, cite `[n]` inline, frame
  medications as clinician-discussion options (never a prescription/dose), fixed
  headings + safety disclaimer, temperature 0.2.
- **`/recommend` endpoint** takes patient fields + `comorbidities: list[str]`, predicts
  the stage, then generates. `generate.py` split into `generate_from_chunks` (explicit
  context) + `generate_recommendations` (retrieve then generate) so evaluation can feed
  a specific retrieval set.
- **UI:** after the prediction, a comorbidity step (HF, CKD, ASCVD, stroke, hypertension,
  obesity as Yes/No) → `/recommend` → rendered recommendations + sources. Comorbidity
  "Yes" answers become BM25 tokens that pull the right guideline chunks (e.g. HF/CKD →
  SGLT2i/finerenone) — verified end-to-end.

---

## 7. Validation with RAGAS (in progress — hit a hardware wall)

**Intent:** quantify faithfulness, answer relevancy, context precision, and context
recall, fully local (Ollama judge + local embeddings), and use it to *prove* the design
choices (hybrid vs dense).

**What we found (all hardware/tooling reality, documented so we don't repeat it):**
- **RAGAS + langchain pinning:** ragas 0.4.3 declares langchain deps with no upper
  bound, so uv pulled langchain 1.x, which removed a `vertexai` shim ragas still imports.
  Fixed by pinning the langchain stack to `>=0.3,<1.0` in the `eval` dependency group.
  Also needed `rapidfuzz` for the testset transforms.
- **Judge model: qwen3-vl:8b → qwen2.5vl:3b.** The 8B judged at **~245s/call** (6 GB
  doesn't fit GPU memory → CPU fallback), hitting RAGAS's 180s timeout → `nan`. The 3B
  fits and runs ~1.5s/call. Judge is env-overridable (`JUDGE_MODEL`) for bigger hardware.
- **Synthetic testset generation crashed the Ollama runner** ("model runner has
  unexpectedly stopped ... resource limitations") — `TestsetGenerator` fires many
  concurrent large-context calls. **Pivoted to a hand-authored 12-question gold set**
  (`rag/eval/testset.json`) grounded in the ADA guidelines — better for a medical eval
  anyway, and it sidesteps the crash. The synthetic script (`gen_testset.py`) remains for
  future use on stronger hardware.
- **The evaluation itself is impractically slow locally:** a 3-question run took **~9
  hours** (mostly blocked on Ollama, with `OUTPUT_PARSING_FAILURE` retries because the
  small judge emits malformed JSON). The low faithfulness/relevancy numbers from that run
  are **judge artifacts, not real quality signals**.

**Conclusion / open decision:** fully-local RAGAS validation is not viable on this
machine. Realistic paths forward:
1. **Cloud judge for evaluation only** — the eval questions are guideline-based with no
   patient PII, so using an API judge for scoring doesn't compromise production privacy;
   it's fast and emits reliable structured output. (Production inference stays 100% local.)
2. **Lightweight custom retrieval metrics** — e.g. embedding hit-rate / MRR of retrieved
   chunks against the gold references (no LLM judge needed), which still validates the
   dense-vs-hybrid choice cheaply.
3. Accept long overnight runs (not recommended given ~9h/3 questions).

**Eval scaffolding built (kept):** `rag/eval/ragas_local.py` (local judge + embeddings
wrappers + RunConfig), `gen_testset.py`, `run_eval.py` (retrieve→generate→score, with
`--dense-only` and `--limit`), `testset.json` (gold set, later expanded to 43 questions
with `reference_contexts`).

### Retrieval evaluation — the result we did get (semantic, no LLM judge)

The 43-question gold set includes `reference_contexts`, which enabled a fast retrieval
eval that sidesteps the slow judge entirely. First attempt used RAGAS's **non-LLM**
context metrics (string-distance) → near-zero, misleading scores: our chunks carry
heading prefixes / different whitespace than the reference excerpts, so a clearly
relevant chunk (cosine 0.77 to the reference) still string-matches ~0. **Switched to
semantic matching** with the index's own embeddings (`rag/eval/retrieval_eval.py`):
coverage, recall@k, hit@k, MRR via cosine similarity between retrieved chunks and
reference contexts.

**Result (k=6, threshold 0.7): hybrid retrieval beats dense.**

| | coverage | recall@k | hit@k | MRR |
|---|---|---|---|---|
| hybrid | 0.795 | **0.884** | 0.884 | **0.707** |
| dense  | 0.794 | 0.837 | 0.837 | 0.611 |
| hybrid — medication (n=22) | 0.812 | **0.955** | 0.955 | **0.701** |
| dense — medication (n=22)  | 0.809 | 0.909 | 0.909 | 0.649 |

Coverage is ~equal (both find similar content), but hybrid wins on **recall and MRR** —
it ranks the relevant chunk higher — and the gap is largest on **medication** questions,
exactly where BM25's exact-token matching is meant to help. This is the quantitative
confirmation of the hybrid + RRF design choice.

**LLM-judged cross-check** (`context_eval_llm.py`, Claude as judge — direct calls, since
RAGAS deadlocks with Anthropic): context recall 0.816, avg-precision 0.790, simple
precision@6 0.471. The **recall agrees with the embedding method** (0.82 vs 0.88) — two
independent signals validating that retrieval fetches the needed content. Precision@6
looks low, but that's the **multi-query fan-out by design**: 6 diverse chunks (lifestyle +
medication + per-comorbidity) means ~half are off-topic *for a narrow eval question* yet
wanted for a full recommendation — and avg-precision (0.79) >> simple-P@6 (0.47) confirms
the relevant chunks rank at the top. Medication pillar is strong by both judges (LLM P 0.85,
R 0.90).

### Generator evaluation — Tier 1 (no LLM judge)

`rag/eval/generator_eval.py` generates an answer per question (llama3.1) and scores two
things cheaply: (1) answer semantic similarity to the reference, (2) a drug-grounding /
hallucination check — every specific medication named must be supported by the retrieved
context (agent name or its class token present). It also saves the retrieved context +
answer per question to `generator_results.json` (otherwise the retrieved context is
computed live and never persisted).

**What it found & fixed:**
- **Doses:** 0 real drug doses (an initial "10" was a regex false-positive matching lab
  values like `126 mg/dL`; fixed to require a drug unit not followed by `/`).
- **Drug hallucination (real bug):** the generator always emitted a Medication section, so
  on questions with no medication context (diagnosis, CGM, bariatric surgery) it invented
  drug names from parametric knowledge — **9 true hallucinations / 20 mentions (grounding
  0.45)**. Strengthened the prompt ("never introduce a drug not in the CONTEXT; if none,
  say so and name no drugs"). Re-run: mentions 20→13, **hallucinations 9→3, grounding
  0.45→0.77**, still 0 doses. Residual 3 is the ceiling of prompt-only mitigation on an 8B
  local model, so we added a **deterministic drug-grounding safety filter** in
  `generate.py` (`apply_drug_safety`): after generation, any named agent not supported by
  the retrieved context (neither the agent nor its class token present) is redacted before
  the text reaches the user — ungrounded drugs reaching the user is now **0 by
  construction**, independent of the model. `generate_from_chunks` returns both the
  filtered `recommendations` (user-facing) and `raw_recommendations` (so the eval still
  measures the model's true rate) plus `redacted_drugs`.
- **Answer similarity ~0.46:** uniform across pillars → it mostly measures format distance
  (verbose recommendations vs terse references), not correctness. Weak proxy; a
  patient-scenario testset would measure the generator's real job better.

### Generator evaluation — Tier 2 (LLM-judged, Claude Sonnet 4.6)

Judge = **Claude Sonnet 4.6** via the Anthropic API (different family from the llama3.1
generator → no self-bias; eval data has no patient PII; embeddings stay local). API key
loaded from a git-ignored `.env`.

**A real fight with RAGAS:** RAGAS's `evaluate()` (and even a single metric's
`single_turn_ascore`) **deadlocks on async Anthropic calls** with our pinned langchain
0.3.x stack — a single metric hung ~60s/call and timed out to `nan`, though a direct
`ChatAnthropic` call (sync *and* async) returns in ~1s. Capping retries didn't help; it's
an executor/event-loop incompatibility, and we can't unpin langchain (ragas 0.4.3 imports
a `vertexai` shim removed in langchain 1.x). **Pivoted** to `rag/eval/judge_direct.py`:
computes the two standard metrics with direct Claude calls (faithfulness = fraction of the
answer's claims supported by context; answer_relevancy = 0-1 rating), async with a
concurrency cap. Fast and reliable.

**Results (43 questions): faithfulness 0.866, answer_relevancy 0.450 overall.** By pillar:

| pillar | faithfulness | relevancy | n |
|---|---|---|---|
| medication | 0.921 | 0.436 | 22 |
| diet | 0.952 | 0.850 | 3 |
| lifestyle | 1.000 | 0.917 | 3 |
| classify | 0.621 | 0.286 | 7 |
| screening | 0.778 | 0.333 | 3 |
| safety | 0.833 | 0.200 | 3 |

**Interpretation:** on the pillars the system is designed for (medication/diet/lifestyle),
faithfulness is 0.92-1.0 and relevancy is strong for diet/lifestyle. The averages are
dragged down by classify/screening/safety — terse factual Q&A that a *recommendation*
generator answers with advice rather than a direct fact, so low relevancy + lower
faithfulness there. **The low overall relevancy is a testset↔generator mismatch, not a
defect;** a patient-scenario testset would measure the generator's real job. Combined with
Tier 1 (medication faithfulness 0.92 + the deterministic filter → 0 ungrounded drugs reach
the user), the generation safety story is solid.

### Generator A/B — llama3.1 vs Claude Sonnet 4.6

The generator was made swappable (`GENERATOR_MODEL` env; `_chat()` routes Ollama or the
Anthropic API). `compare_generators.py` regenerated all 43 answers with each model through
the *current* pipeline and judged both with the same Claude faithfulness/relevancy harness:

| | Faithfulness | Relevancy |
|---|---|---|
| llama3.1 (local) | 0.878 | 0.508 |
| **Claude Sonnet 4.6** | **0.987** | **0.808** |

Claude wins everywhere — near-perfect grounding, **1.00 medication faithfulness** (vs 0.87),
and it fixes the weak relevancy pillars (classify 0.27→0.86, safety 0.13→0.60), showing that
part of the earlier relevancy confound was a model-capability gap, not just format.

**The decision is NOT purely quality — it's quality vs. privacy.** The project's founding
rationale was "local LLM so no patient data leaves the machine." `/recommend` sends the
patient profile (labs, comorbidities) to the generator, so switching to Claude means
**patient data leaves the machine** (to Anthropic). llama3.1 keeps everything local. So:
Claude for max quality where cloud PHI is acceptable; llama3.1 to preserve the local-only
guarantee. Kept llama3.1 as the default; Claude is opt-in via `GENERATOR_MODEL`.

**Eval scripts:** `retrieval_eval.py` (semantic hybrid-vs-dense), `generator_eval.py`
(Tier 1: similarity + drug-grounding), `judge_direct.py` (Tier 2: faithfulness +
relevancy via Claude). `run_eval.py`/`judge_eval.py` remain but hit the RAGAS async
deadlock with Anthropic.

---

## Cross-cutting principles that kept recurring

- **One interpreter, one lockfile** (uv + pyproject) — most early pain was version drift.
- **Python 3.14 is bleeding-edge:** repeatedly bit us on wheels/compat (onnxruntime,
  faiss, the langchain/ragas pins). Prefer libs with cp314 wheels; isolate risky ones.
- **Match the tool to the data, not the hype:** pymupdf4llm over markitdown, hybrid over
  dense, semantic over fixed chunks — each driven by inspecting the actual output.
- **Keep transforms in one place** so training and serving (and dense vs hybrid eval)
  stay identical.
