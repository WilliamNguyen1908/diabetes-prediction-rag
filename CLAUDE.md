# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Build an application that predicts whether a person has diabetes using a machine-learning model, exposed through a web UI for patient input. After a prediction is made, a RAG system generates personalized health recommendations based on the patient's input **and** the model's result.

## Intended workflow (end-to-end)

1. **Patient input** — user fills out a web form (vitals/labs/lifestyle) in the browser.
2. **Preprocessing** — the backend cleans and encodes the input to match the features the model was trained on.
3. **Prediction** — the trained ML model returns a diabetes class/probability.
4. **Retrieval (RAG)** — the patient profile + prediction are turned into a query; relevant guidance is retrieved from a vector store built over a curated knowledge base (diet, exercise, clinical guidelines).
5. **Generation** — a local LLM produces tailored recommendations grounded in the retrieved context, conditioned on the prediction.
6. **Response** — prediction + recommendations are rendered back in the UI.

Keep these as distinct, independently testable stages. The trained model artifact and the vector index are build outputs — regenerate them from scripts, do not hand-edit.

## Project status

**Built and working end-to-end.** The full pipeline exists: XGBoost prediction ([predict.py](predict.py)) behind a FastAPI + Jinja2 UI ([app.py](app.py), [templates/](templates/)), and the complete RAG recommendation system ([rag/](rag/)) — PDF ingestion, semantic chunking, hybrid retrieval + rerank, corrective retrieval, grounded generation (llama3.1 or Claude), drug-safety redaction, runtime NLI grounding, input/output guardrails, SQLite audit logging, and an evaluation suite ([rag/eval/](rag/eval/)). Runs on **Python 3.14** via `uv`. The pipeline is mapped stage-by-stage below (build-time + query-time), with the "chose → failed → changed" history and a function reference.

## Tech stack

Python 3.14, managed with `uv` (`pyproject.toml` + `uv.lock`; `requirements.txt` mirrors pinned versions).

### Prediction model
- `pandas`, `numpy`, **scikit-learn** preprocessing, **XGBoost** classifier.
- Artifacts: model in [diabetes_xgb.json](diabetes_xgb.json), preprocessing (one-hot + scaler + metadata) in [preprocess_bundle.joblib](preprocess_bundle.joblib). Live inference reuses the exact training transforms via [predict.py](predict.py).
- Exploratory work in [model.ipynb](model.ipynb).

### Web application / UI
- **FastAPI** backend serving **HTML via Jinja2 templates** (server-rendered form + results page).
- **Pydantic** models for request validation — important for clamping/validating medical inputs.
- Run with **uvicorn**.

### RAG recommendation system
- **Local / open LLM via Ollama** (`llama3.1`) for generation by default — no patient data leaves the machine — **swappable to `claude-sonnet-4-6`** (Anthropic) via `GENERATOR_MODEL`. Claude scores clearly better (see eval) but sends PHI to the cloud, so llama3.1 stays the privacy-preserving default.
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim, normalized) for embeddings — resolves cleanly on Python 3.14 (torch 2.12).
- **Vector store: brute-force cosine over a NumPy matrix, not FAISS.** The corpus is tiny (13 PDFs → ~1,300 chunks); a matrix dot product is instant and avoids faiss-cpu's missing cp314 wheels. Swap-in point is isolated in [rag/store.py](rag/store.py).
- **Retrieval is hybrid: dense + BM25 fused with RRF** ([rag/retrieve.py](rag/retrieve.py)). Dense (cosine over embeddings) handles semantics; **BM25** (`rank-bm25`) handles exact tokens — drug names (`empagliflozin`, `SGLT2`, `GLP-1`), dosages, comorbidity terms (`heart failure`, `stroke`) — which the medication-recommendation flow depends on. **Reciprocal Rank Fusion** combines the two rank lists without reconciling their score scales (`1/(rrf_k + rank)`, `rrf_k=60`). BM25 tokenizer keeps internal hyphens/digits so `glp-1`/`a1c`/`sglt2` survive. `HybridRetriever` loads the index + builds BM25 once at construction — instantiate one at app startup and reuse.
- **PDF → Markdown: `pymupdf4llm`, not markitdown.** markitdown's pdfminer backend jumbles these multi-column ADA journal PDFs (interleaved columns, no headings); pymupdf4llm is layout-aware and recovers reading order, paragraphs, headings, and tables. (markitdown was tried first — it also needs a `magika`/`onnxruntime` override to install on 3.14 — and removed.)

**Ingestion pipeline** (`rag/`, all one-time build scripts; index is a git-ignored build output):
- Source PDFs live in `knowledge/pdfs/`; generated `knowledge/md/` and `knowledge/index/` are git-ignored.
- [rag/convert.py](rag/convert.py) — `knowledge/pdfs/*.pdf` → `knowledge/md/*.md` (pymupdf4llm).
- [rag/chunk.py](rag/chunk.py) — **hybrid semantic chunker** (medical context matters, so fixed-size windows are avoided): split on Markdown headings, keep the heading trail (`H1 > H2 > H3`) as context on every chunk, and within oversized sections split at sentence-embedding **semantic breakpoints** (percentile of consecutive cosine distance) with a hard char cap. Tables kept intact; `References`/front-back-matter sections dropped.
- [rag/embed.py](rag/embed.py) — one cached embedder shared by chunker, ingest, and (later) retriever.
- [rag/ingest.py](rag/ingest.py) — md → chunks → embeddings → `knowledge/index/{embeddings.npy,chunks.json}` via [rag/store.py](rag/store.py).
- Build: `uv run python rag/convert.py` then `uv run python rag/ingest.py`. Re-run only when the PDFs change.

**Generation** ([rag/generate.py](rag/generate.py)): **llama3.1 via Ollama** (needs the Ollama server running; `ollama` Python client). **Personalization:** `summarize_patient()` turns the raw form input into a clinically-flagged PATIENT PROFILE (each value labelled normal/high/low vs. standard reference ranges — BMI, HbA1c, glucose, full lipid panel, BP staging, etc. — plus a "Notable abnormal findings" line); the prompt requires the model to cite the patient's own numbers ("Given your LDL of 160 mg/dL (high)…"). `/recommend` passes the full patient dict. Retrieval sub-queries: one lifestyle, one general medication, **plus one medication query per comorbidity** (so drug-specific chunks — e.g. the agents that reduce stroke risk, or SGLT2i/finerenone for HF/CKD — actually reach the context) → merged/de-duped (cap 10) → grounded prompt. The system prompt forces the model to use ONLY the provided context and cite `[n]`, and for **Medication Considerations to be specific**: name the relevant classes + representative example agents *that appear in the context* (e.g. "SGLT2 inhibitors (such as empagliflozin)") with a per-patient rationale — but **not** invent agents or give doses/titration (clinician's decision), and say so if the context doesn't support a medication. Fixed headings + safety disclaimer, temperature 0.2. The `HybridRetriever` is cached (`lru_cache`) so it loads once.

**Reranking, corrective retrieval, safety** (all in [rag/retrieve.py](rag/retrieve.py) / [rag/generate.py](rag/generate.py) / [rag/grounding.py](rag/grounding.py) / [rag/validation.py](rag/validation.py)):
- **Cross-encoder final rerank** — after RRF fuses the merged pool, a cross-encoder (`BAAI/bge-reranker-base`) rescores it against one combined patient query → top-12 (classic retrieve-then-rerank; toggle `FINAL_RERANK`).
- **Corrective retrieval (CRAG)** — if a comorbidity has no supporting chunk, the LLM rewrites that sub-query and retries once.
- **Drug-grounding safety** — deterministically **redacts any drug name not present in the retrieved context** (hard guarantee: no ungrounded drug reaches the user).
- **Runtime NLI grounding** — per-claim DeBERTa entailment vs context + patient profile flags unsupported claims (advisory; toggle `GROUNDING_NLI`).
- **Input/output guardrails** — `validate_input` rejects contradictory input (e.g. stroke=Yes with cardiovascular history=No) and impossible ranges *before* generation; `check_output` verifies headings/disclaimer, no doses, resolvable citations.

**Endpoint** ([app.py](app.py)): `POST /recommend` takes the patient fields + a `comorbidities: list[str]`, validates, predicts the stage, then generates. Returns `{stage, probabilities, recommendations, sources, warnings}`. `rag/` is put on `sys.path` so app.py imports the flat rag modules (`from generate import ...`); `generate` is imported lazily inside the handler so app startup doesn't load torch/the LLM. First `/recommend` call is slow (model load); subsequent calls are just llama3.1 inference (~10–20s). Each call is audit-logged to SQLite ([db.py](db.py)).

## Pipeline map — raw data → user response

End-to-end map of every stage, the function/file that handles it, and — where it matters — **what we chose first, why it failed, and what we changed**. Two pipelines: **A. build-time** (offline, PDFs → index; run once, re-run when PDFs change) and **B. query-time** (online, patient form → recommendation).

Note on "bi-encoder then cross-encoder": the **bi-encoder** (all-MiniLM) is the *dense half of retrieval* (embeds query & chunks separately — fast, wide net); the **cross-encoder** (bge-reranker) is the *final filter* (reads query+chunk together — slow, precise).

### A. Build-time  `knowledge/pdfs/*.pdf → knowledge/index/`

```
PDFs ──convert──> Markdown ──chunk──> chunks ──embed──> vectors ──ingest/store──> index
```

| # | Stage | File · function | What / why |
|---|---|---|---|
| 1 | **PDF → Markdown** | [rag/convert.py](rag/convert.py)`::main` (pymupdf4llm) | Layout-aware extraction of the 13 ADA guideline PDFs. |
| 2 | **Semantic chunking** | [rag/chunk.py](rag/chunk.py)`::chunk_markdown` (+ `_iter_sections`, `_segments`, `_semantic_groups`) | Split on Markdown headings, keep the `H1 > H2 > H3` trail as context, semantic-split oversized sections at sentence-embedding breakpoints. Drops References/front-matter. |
| 3 | **Embedding** | [rag/embed.py](rag/embed.py)`::embed`/`get_model` | `all-MiniLM-L6-v2`, 384-dim, normalized. One cached model shared by chunker, ingest, retriever. |
| 4 | **Index build** | [rag/ingest.py](rag/ingest.py)`::main` → [rag/store.py](rag/store.py)`::save_index` | md → chunks → embeddings → `knowledge/index/{embeddings.npy, chunks.json}`. |
| — | **Vector store** | [rag/store.py](rag/store.py) (`load_index`, `search`) | Brute-force cosine over a NumPy matrix. |

**Chose → failed → changed (build-time):**
- **PDF converter: markitdown → pymupdf4llm.** markitdown's pdfminer backend jumbled the multi-column journal PDFs (interleaved columns, no headings); pymupdf4llm recovers reading order, headings, tables (52 headings vs 0 on one doc). Also dropped markitdown's `magika`/`onnxruntime` install headache on Python 3.14.
- **Chunking: fixed-size → semantic.** Fixed windows cut a dose from its condition — unsafe for medical text. Switched to heading-structure + embedding-breakpoint chunks (~1,300).
- **Vector store: FAISS → NumPy cosine.** faiss-cpu has no Python 3.14 wheel and the corpus is tiny (a matrix dot product is instant). Swap point isolated in `store.py`.

### B. Query-time  `patient form → recommendation`

```
form ─predict─> stage ─validate─> profile ─decompose─> sub-queries
   ─hybrid retrieve─> ─RRF merge─> ─corrective retrieve─> ─cross-encoder rerank─> top-12
   ─generate(LLM)─> ─drug-redact─> ─NLI grounding─> ─output checks─> response
```

| # | Stage | File · function | What / why |
|---|---|---|---|
| 1 | **HTTP entry** | [app.py](app.py)`::recommend` (`POST /recommend`) | Takes patient fields + `comorbidities[]`. |
| 2 | **Prediction** | [predict.py](predict.py)`::predict_patient` (+ `preprocess_patient`) | XGBoost → stage + probabilities. |
| 3 | **Input guardrails** | [rag/validation.py](rag/validation.py)`::validate_input` | Reject contradictions & impossible ranges; warn on soft issues. Runs *before* generation. |
| 4 | **Patient profiling** | [rag/generate.py](rag/generate.py)`::summarize_patient` | Raw values → flagged clinical PROFILE (normal/high/low vs reference ranges) for personalization. |
| 5 | **Query decomposition** | [rag/generate.py](rag/generate.py)`::build_queries` (+ `_canonical`) | Stage-aware base queries + one medication query per comorbidity (`stroke → ASCVD` etc.), deduped. |
| 6 | **Hybrid retrieval** | [rag/retrieve.py](rag/retrieve.py)`::HybridRetriever.search` (+ `tokenize`, `rrf_fuse`) | Per sub-query: dense cosine + BM25, fused with **RRF**. |
| 7 | **Merge / dedup** | [rag/generate.py](rag/generate.py)`::retrieve_context` (`add_hits`) | Pool all sub-query hits, dedup by (source, text). |
| 8 | **Corrective retrieval** | `retrieve_context` + `_rewrite_query` + `validation.check_retrieval_coverage` | If a comorbidity has no supporting chunk, LLM rewrites the query and retries once. |
| 9 | **Final-pool rerank** | [rag/retrieve.py](rag/retrieve.py)`::rerank_chunks` (bge-reranker) vs `_combined_query` | Cross-encoder rescores the merged pool → top-12. Toggle `FINAL_RERANK`. |
| 10 | **Generation** | [rag/generate.py](rag/generate.py)`::generate_from_chunks` → `_chat` | Grounded, personalized prompt → llama3.1 or claude-sonnet-4-6 via `GENERATOR_MODEL`. |
| 11 | **Drug-grounding safety** | [rag/generate.py](rag/generate.py)`::apply_drug_safety` | Deterministically redact any drug name not in the retrieved context. |
| 12 | **Runtime NLI grounding** | [rag/grounding.py](rag/grounding.py)`::check_grounding` (DeBERTa) | Per-claim entailment tripwire vs context + profile; flags unsupported claims (advisory). |
| 13 | **Output structure checks** | [rag/validation.py](rag/validation.py)`::check_output` | Headings/disclaimer present, no doses, citations resolve. |
| 14 | **Response** | [app.py](app.py)`::recommend` | `{stage, probabilities, recommendations, sources, warnings}` + audit log ([db.py](db.py)). |
| 15 | **UI render** | [templates/index.html](templates/index.html) | Form → prediction bars → comorbidity step → recommendations + sources. |

**Chose → failed → changed (query-time):**
- **Retrieval: dense-only → hybrid (dense + BM25 + RRF).** Dense embeddings under-weight exact tokens; drug names (`empagliflozin`, `SGLT2`, `finerenone`) need lexical matching. Hybrid > dense on recall/MRR, biggest gap on medication questions.
- **Queries: blind `"{stage} diabetes"` splice → stage-aware decomposition.** The naive version produced `"No Diabetes diabetes"` and pulled treatment chunks for non-diabetics. Now prevention/screening for low-risk, treatment for Type 2.
- **Reranking: none → cross-encoder final filter.** Added `rerank_chunks` (ms-marco first, then bge-reranker-base — better MRR). Reranking trades a little recall for top ranking, so it's toggleable (`FINAL_RERANK`) and runs on the *merged pool*, not per-query.
- **Generator: local llama3.1 default; Claude opt-in.** Claude is clearly better (A/B below) but `/recommend` sends patient data to the generator, so Claude means PHI leaves the machine. llama3.1 is the privacy-preserving default.
- **Safety added incrementally** after observing failures: invented drug names → deterministic drug redaction; ~13% of claims not fully grounded → runtime NLI tripwire; contradictory input → validation.

## Evaluation  `rag/eval/`

| What | File | Result |
|---|---|---|
| Retrieval (embedding) | `retrieval_eval.py` | hybrid recall 0.88 / MRR 0.71 > dense 0.84 / 0.61 |
| Retrieval (LLM-judged) | `context_eval_llm.py` | recall 0.82 (cross-validates the embedding number) |
| Generation Tier-1 (no judge) | `generator_eval.py` | 0 fabricated doses; drug-grounding + redaction |
| Generation Tier-2 (Claude judge) | `judge_direct.py` | faithfulness 0.99 (Claude) vs 0.88 (llama) |
| Generator A/B | `compare_generators.py` | Claude wins: faith 0.99 / rel 0.81 vs 0.88 / 0.51 |
| Visualization | `result.py` | `results.png` |

**Chose → failed → changed (eval):**
- **Local RAGAS judge → direct-Claude judge.** The 8B local judge ran ~245s/call (infeasible), the 3B was noisy, and RAGAS's `evaluate()` deadlocks on the async Anthropic client. Now faithfulness/relevancy come from direct Claude calls (`judge_direct.py`).
- **Synthetic testset → hand-authored gold set.** RAGAS's `TestsetGenerator` crashed the Ollama runner; a curated 43-question set with `reference_contexts` replaced it.
- **RAGAS non-LLM context metrics → semantic matching.** String-distance scored relevant chunks ~0 (formatting differs); switched to embedding-similarity in `retrieval_eval.py`.

## Config knobs (env vars)

| Var | Default | Effect |
|---|---|---|
| `GENERATOR_MODEL` | `llama3.1` | `claude-sonnet-4-6` to generate via Anthropic (needs `ANTHROPIC_API_KEY` in gitignored `.env`). |
| `FINAL_RERANK` | `1` | `0` to skip the cross-encoder rerank (RRF order instead). |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | swap the cross-encoder. |
| `GROUNDING_NLI` | `1` | `0` to skip the runtime NLI grounding check. |
| `JUDGE_MODEL` | `claude-sonnet-4-6` | judge model for the eval scripts. |

## Function reference — purpose + input/output types

Recurring type: a **`chunk`** is a `dict` with keys `text` (str), `source_file` (str), `heading` (str). After retrieval it also carries score keys (`rrf_score`, `dense_score`, `bm25_score`, and `rerank_score` if reranked). `list[chunk]` means a Python list of those.

### [predict.py](predict.py) — ML prediction
| Function | What it does | Input → Output |
|---|---|---|
| `preprocess_patient(raw)` | Raw patient values → the model's scaled feature row (one-hot + scaler) | `dict` → `pandas.DataFrame` (1×44) |
| `predict_patient(raw)` | Predict the diabetes stage and class probabilities | `dict` → `tuple[str, dict[str,float]]` |

### [rag/convert.py](rag/convert.py) — PDF → Markdown (build-time)
| Function | What it does | Input → Output |
|---|---|---|
| `main()` | Convert every `knowledge/pdfs/*.pdf` to `knowledge/md/*.md` | `()` → `None` (writes files) |

### [rag/chunk.py](rag/chunk.py) — semantic chunking (build-time)
| Function | What it does | Input → Output |
|---|---|---|
| `chunk_markdown(text, source_file)` | One Markdown doc → list of retrieval chunks | `(str, str)` → `list[chunk]` |
| `_iter_sections(text)` | Yield `(heading_trail, body)` per heading section, dropping noise | `str` → `generator[tuple[str, str]]` |
| `_segments(body)` | Split a section into atomic segments (sentences; tables/lists kept whole) | `str` → `list[str]` |
| `_semantic_groups(segs)` | Group segments into chunk-sized strings at size/semantic breakpoints | `list[str]` → `list[str]` |
| `_hard_wrap(seg)` | Char-split an oversized single segment | `str` → `list[str]` |
| `_is_skippable(heading_trail)` / `_is_noise(line)` | Drop References/front-matter / running-header noise | `str` → `bool` |
| `_clean_heading(text)` | Strip Markdown emphasis/whitespace from a heading | `str` → `str` |

### [rag/embed.py](rag/embed.py) — embeddings
| Function | What it does | Input → Output |
|---|---|---|
| `get_model()` | Load/cache the sentence-transformers model | `()` → `SentenceTransformer` |
| `embed(texts)` | Embed strings → L2-normalized vectors | `list[str]` → `np.ndarray` (n×384, float32) |

### [rag/ingest.py](rag/ingest.py) / [rag/store.py](rag/store.py) — index build + vector store
| Function | What it does | Input → Output |
|---|---|---|
| `ingest.main()` | md → chunks → embeddings → save the index | `()` → `None` (writes files) |
| `save_index(embeddings, chunks)` | Persist `embeddings.npy` + `chunks.json` | `(np.ndarray, list[chunk])` → `None` |
| `load_index()` | Load the index into memory | `()` → `tuple[np.ndarray, list[chunk]]` |
| `search(query_vec, embeddings, chunks, k)` | Cosine top-k over the matrix | `(np.ndarray, np.ndarray, list[chunk], int)` → `list[chunk]` (+`score`) |

### [rag/retrieve.py](rag/retrieve.py) — hybrid retrieval + rerank
| Function | What it does | Input → Output |
|---|---|---|
| `tokenize(text)` | BM25 tokenizer (keeps `glp-1`, `sglt2`, digits) | `str` → `list[str]` |
| `rrf_fuse(rank_lists, rrf_k)` | Reciprocal Rank Fusion of ranked index lists | `list[list[int]]` → `list[tuple[int,float]]` |
| `get_reranker()` | Load/cache the cross-encoder reranker | `()` → `CrossEncoder` |
| `rerank_chunks(query, chunks, top_k)` | Cross-encoder rescore a pool → top_k reordered | `(str, list[chunk], int)` → `list[chunk]` (+`rerank_score`) |
| `HybridRetriever.__init__()` | Load index + build BM25 once | `()` → instance |
| `HybridRetriever.search(query, k, …)` | Dense + BM25 → RRF (→ optional per-query rerank) → top-k | `str` (+ ints/bools) → `list[chunk]` |

### [rag/generate.py](rag/generate.py) — query building, retrieval orchestration, generation
| Function | What it does | Input → Output |
|---|---|---|
| `summarize_patient(patient)` | Raw values → flagged clinical PROFILE text | `dict` → `str` |
| `_canonical(comorbidity)` | Map a comorbidity label to its canonical key (`stroke`→`ASCVD`) | `str` → `str | None` |
| `build_queries(stage, comorbidities)` | Form the stage-aware + per-comorbidity sub-queries | `(str, list[str])` → `list[tuple[str,str]]` |
| `_combined_query(stage, comorbidities)` | One representative query for the final rerank | `(str, list[str])` → `str` |
| `_rewrite_query(topic, stage)` | LLM rewrite of an uncovered query (CRAG) | `(str, str)` → `str | None` |
| `retrieve_context(stage, comorbidities)` | Full retrieval: fan-out → merge → corrective → rerank | `(str, list[str])` → `tuple[list[chunk], list[tuple]]` |
| `_format_context(chunks)` | Chunks → numbered `[n] (source)…` context string | `list[chunk]` → `str` |
| `_chat(system, user, …)` | One LLM completion, routed to Ollama or Claude | `(str, str)` → `str` |
| `find_drug_agents(text)` | Drug names present in a text | `str` → `list[str]` |
| `ungrounded_agents(text, context_lower)` | Drugs named but not supported by context | `(str, str)` → `list[str]` |
| `apply_drug_safety(text, chunks)` | Redact ungrounded drug names | `(str, list[chunk])` → `tuple[str, list[str]]` |
| `generate_from_chunks(chunks, stage, …)` | Prompt → LLM → safety filters → result | `(list[chunk], str, …)` → `dict` |
| `generate_recommendations(stage, comorbidities, patient, …)` | Retrieve + generate; top-level entry | `(str, list[str], dict)` → `dict` |

`generate_*` return `dict` keys: `recommendations` (str), `sources` (list[dict]), `warnings` (list[str]), `redacted_drugs` (list[str]), `ungrounded_claims` (list[str]), `queries`, `retrieved` (list[dict]).

### [rag/grounding.py](rag/grounding.py) — runtime NLI grounding
| Function | What it does | Input → Output |
|---|---|---|
| `_claims(text)` | Split answer into checkable claim sentences | `str` → `list[str]` |
| `check_grounding(text, chunks, …)` | Per-claim NLI entailment vs context+profile | `(str, list[chunk])` → `dict{claims, ungrounded}` |

### [rag/validation.py](rag/validation.py) — guardrails
| Function | What it does | Input → Output |
|---|---|---|
| `validate_input(patient, comorbidities, stage)` | Input consistency + range checks | `(dict, list[str], str)` → `dict{errors, warnings}` |
| `check_retrieval_coverage(chunks, stage, comorbidities)` | Comorbidities with no supporting chunk | `(list[chunk], str, list[str])` → `list[str]` |
| `check_output(text, n_sources)` | Headings/disclaimer/dose/citation checks | `(str, int)` → `list[str]` |

### [db.py](db.py) — audit log
| Function | What it does | Input → Output |
|---|---|---|
| `init_db()` | Create the SQLite log table if absent | `()` → `None` |
| `log_recommendation(patient, stage, comorbidities, queries, retrieval)` | Insert one audit row | `(dict, str, list[str], list[dict], list[dict])` → `int` |

### [app.py](app.py) — FastAPI endpoints
| Function | What it does | Input → Output |
|---|---|---|
| `index(request)` | Render the patient form | `Request` → `HTMLResponse` |
| `predict(patient)` | `POST /predict` — stage + probabilities | `Patient` → `dict` |
| `recommend(req)` | `POST /recommend` — validate → generate → log | `RecommendRequest` → `dict` |

## Building the model

The model is already trained (artifacts above); this section documents the cleaning approach used, for reference when retraining. Keep each step as a reusable function so the same transforms can be reapplied to live UI input at prediction time.

### 1. Data cleaning

**General approach:**
- **Profile before touching anything** — inspect shape, dtypes, per-column missing counts, duplicate rows, and unique values / value counts for categoricals. Let the profile drive the cleaning, don't assume.
- **Fix column names and types** — strip whitespace, lowercase, snake_case headers; coerce numeric columns that loaded as `object` with `pd.to_numeric(..., errors="coerce")` so junk becomes `NaN`.
- **Normalize categorical text** — strip whitespace and unify casing (this is the root cause of the `"N "` vs `"N"` and `F`/`f` issues). Map to a canonical set of categories; treat anything outside it as missing.
- **Drop identifier / non-predictive columns** so the model can't memorize them (see per-dataset notes).
- **Duplicates** — decide exact vs. near duplicates; drop exact duplicate rows.
- **Missing values** — quantify first, then choose per column: drop columns with excessive missingness; for the rest, impute (median for skewed numerics, mean for roughly symmetric, mode/`"unknown"` category for categoricals). **Fit imputers on the training split only** and apply to validation/test to avoid leakage. Prefer `sklearn` imputers inside the pipeline over ad-hoc `fillna` so the same statistics apply at inference.
- **Outliers / impossible values** — range-check physiologically bounded fields (e.g. age, BMI, blood pressure, glucose must be positive and within plausible clinical ranges). Set impossible values to `NaN` (then impute) rather than deleting rows outright.
- **Target cleaning** — normalize label text/casing, confirm the class set, and note class imbalance (address later at the modeling stage, e.g. class weights / resampling — not by editing data here).
- **Split before fitting any statistic** — do the train/test split first, then fit cleaning/imputation/scaling on train only.

**`diabetes.csv` specifics:**
- Drop pure noise / identifier columns: `patient_tracking_id`, `data_entry_notes` (free text), and `experimental_peptide_level` (treat as noise unless shown otherwise).
- **Watch for target leakage.** `diagnosed_diabetes` is the natural target; `diabetes_stage` and `diabetes_risk_score` are essentially derived from the diagnosis — exclude them from features.
- Expect messy categoricals (e.g. `gender`, `smoking_status`, `employment_status`) with inconsistent labels — canonicalize as above.

Keep the cleaning logic in one place so both training and the FastAPI prediction path use identical transforms.

## Environment

- **Run the app:** `uv run uvicorn app:app --reload`, then open http://127.0.0.1:8000
- **RAG (default, local):** needs an Ollama server running with `llama3.1` pulled.
- **RAG (cloud) / eval:** set `GENERATOR_MODEL=claude-sonnet-4-6` and `ANTHROPIC_API_KEY` (in gitignored `.env`).
- **Rebuild the index** (only when the PDFs change): `uv run python rag/convert.py` then `uv run python rag/ingest.py`.
- **Run evals** (from the project root, e.g.): `uv run python rag/eval/retrieval_eval.py` (retrieval), `uv run python rag/eval/judge_direct.py` (generation, needs `ANTHROPIC_API_KEY`), `uv run python rag/eval/compare_generators.py` (A/B). Gold testset: `rag/eval/testset.json`.
- **Retrieval smoke test:** `uv run python rag/retrieve.py` (demo harness in its `__main__` block).
- Dependencies are managed with `uv` (`pyproject.toml` + `uv.lock`); `requirements.txt` mirrors the pins. There is no test suite; the eval scripts above are the verification path.

## Dataset

### `diabetes.csv` (~16 MB) — the training source
Large synthetic/survey-style dataset with 35 columns spanning demographics, lifestyle, vitals, and labs (e.g. `age, gender, ethnicity, ..., glucose_fasting, hba1c, diabetes_risk_score, diabetes_stage, diagnosed_diabetes, patient_tracking_id, data_entry_notes, experimental_peptide_level`).
- Intentionally messy: missing values, mixed types, free-text `data_entry_notes`, and noise columns. Inspect and clean before modeling (see leakage notes above).
- Target used: `diabetes_stage` (No Diabetes / Pre-Diabetes / Type 2), predicted from 28 form fields.

Quote paths in shell commands — the project directory name contains a space.
