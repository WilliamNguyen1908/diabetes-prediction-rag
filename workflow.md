# Pipeline Workflow — raw data → user response

End-to-end map of the system, the function/file that handles each stage, and — where it
matters — **what we chose first, why it failed, and what we changed**. Two pipelines:

- **A. Build-time (offline)** — turn source PDFs into a searchable index. Run once (re-run
  when the PDFs change).
- **B. Query-time (online)** — a patient uses the app and gets a grounded recommendation.

## Your mental model vs. what's implemented

| Your model | Implemented as | Note |
|---|---|---|
| Chunking: Semantic Chunking | ✅ hybrid semantic chunker | `rag/chunk.py` |
| (indexing?) | NumPy cosine vector store (not FAISS) | `rag/store.py`, built by `rag/ingest.py` |
| Retrieval: Hybrid Search | ✅ dense (bi-encoder) + BM25, fused with RRF | `rag/retrieve.py` |
| Query Transformation: Query-decomposition | ✅ stage-aware decomposition + corrective rewrite | `rag/generate.py::build_queries`, `_rewrite_query` |
| Re-ranking: bi-encoder → cross-encoder | ✅ this *is* the two-stage: bi-encoder+BM25 retrieve → **cross-encoder** final rerank | `rag/retrieve.py::rerank_chunks` |
| Generator: llama3.1 or sonnet 4.6 | ✅ swappable via `GENERATOR_MODEL` | `rag/generate.py::_chat` |

Small clarification on "bi-encoder then cross-encoder": the **bi-encoder** (all-MiniLM) is
the *dense half of retrieval* (it embeds query & chunks separately — fast, wide net); the
**cross-encoder** (bge-reranker) is the *final filter* (it reads query+chunk together —
slow, precise). Classic **retrieve-then-rerank**.

---

## A. Build-time pipeline (offline)  `knowledge/pdfs/*.pdf → knowledge/index/`

```
PDFs ──convert──> Markdown ──chunk──> chunks ──embed──> vectors ──ingest/store──> index
```

| # | Stage | File · function | What / why |
|---|---|---|---|
| 1 | **PDF → Markdown** | `rag/convert.py::main` (pymupdf4llm) | Layout-aware extraction of the 13 ADA guideline PDFs. |
| 2 | **Semantic chunking** | `rag/chunk.py::chunk_markdown` (+ `_iter_sections`, `_segments`, `_semantic_groups`) | Split on Markdown headings, keep the `H1 > H2 > H3` trail as context, and semantic-split oversized sections at sentence-embedding breakpoints. Drops `References`/front-matter. |
| 3 | **Embedding** | `rag/embed.py::embed` / `get_model` | `all-MiniLM-L6-v2`, 384-dim, normalized. One cached model shared by chunker, ingest, retriever. |
| 4 | **Index build** | `rag/ingest.py::main` → `rag/store.py::save_index` | md → chunks → embeddings → `knowledge/index/{embeddings.npy, chunks.json}`. |
| — | **Vector store** | `rag/store.py` (`load_index`, `search`) | Brute-force cosine over a NumPy matrix. |

**Chose → failed → changed (build-time):**
- **PDF converter: markitdown → pymupdf4llm.** markitdown's pdfminer backend jumbled the
  multi-column journal PDFs (interleaved columns, no headings). pymupdf4llm recovers reading
  order, headings, and tables (52 headings vs 0 on one doc). Also dropped markitdown's
  `magika`/`onnxruntime` install headache on Python 3.14.
- **Chunking: fixed-size → semantic.** Fixed windows cut a dose from its condition — unsafe
  for medical text. Switched to heading-structure + embedding-breakpoint chunks (~1,300).
- **Vector store: FAISS → NumPy cosine.** faiss-cpu has no Python 3.14 wheel, and the corpus
  is tiny (a matrix dot product is instant). Swap point isolated in `store.py`.

---

## B. Query-time pipeline (online)  `patient form → recommendation`

```
form ─predict─> stage ─validate─> profile ─decompose─> sub-queries
   ─hybrid retrieve─> ─RRF merge─> ─corrective retrieve─> ─cross-encoder rerank─> top-12
   ─generate(LLM)─> ─drug-redact─> ─NLI grounding─> ─output checks─> response
```

| # | Stage | File · function | What / why |
|---|---|---|---|
| 1 | **HTTP entry** | `app.py::recommend` (`POST /recommend`) | Takes patient fields + `comorbidities[]`. |
| 2 | **Prediction** | `predict.py::predict_patient` (+ `preprocess_patient`) | XGBoost → stage (No Diabetes / Pre-Diabetes / Type 2) + probabilities. |
| 3 | **Input guardrails (A)** | `rag/validation.py::validate_input` | Reject contradictions (stroke=Yes but CV-history=No) & impossible ranges; warn on soft issues. Runs *before* generation. |
| 4 | **Patient profiling** | `rag/generate.py::summarize_patient` | Raw values → flagged clinical PROFILE (normal/high/low vs reference ranges) for personalization. |
| 5 | **Query decomposition** | `rag/generate.py::build_queries` (+ `_canonical` alias map) | Stage-appropriate base queries + one tailored medication query per comorbidity (`stroke → ASCVD` etc.), deduped. Returns `(label, query)` pairs. |
| 6 | **Hybrid retrieval** | `rag/retrieve.py::HybridRetriever.search` (+ `tokenize`, `rrf_fuse`) | Per sub-query: dense cosine (bi-encoder) + BM25, fused with **RRF**. |
| 7 | **Merge / dedup** | `rag/generate.py::retrieve_context` (`add_hits`) | Pool all sub-query hits, dedup by (source, text). |
| 8 | **Corrective retrieval (#3)** | `retrieve_context` + `_rewrite_query` + `validation.check_retrieval_coverage` | If a comorbidity has no supporting chunk, LLM **rewrites the query and retries once**. |
| 9 | **Final-pool rerank** | `rag/retrieve.py::rerank_chunks` (bge-reranker-base) vs `_combined_query` | **Cross-encoder** rescores the merged pool against one combined patient query → top-12. Toggle `FINAL_RERANK`. |
| 10 | **Generation** | `rag/generate.py::generate_from_chunks` → `_chat` | Grounded, personalized prompt (`SYSTEM_PROMPT`) → **llama3.1 (Ollama)** or **claude-sonnet-4-6 (Anthropic)** via `GENERATOR_MODEL`. |
| 11 | **Drug-grounding safety (C)** | `rag/generate.py::apply_drug_safety` | Deterministically **redact any drug name not in the retrieved context** (hard guarantee). |
| 12 | **Runtime NLI grounding (#4)** | `rag/grounding.py::check_grounding` (DeBERTa NLI) | Per-claim entailment tripwire vs context + patient profile; flags unsupported claims (advisory). |
| 13 | **Output structure checks (F)** | `rag/validation.py::check_output` | Headings/disclaimer present, no doses, citations resolve. |
| 14 | **Response** | `app.py::recommend` | `{stage, probabilities, recommendations, sources, warnings}`. |
| 15 | **UI render** | `templates/index.html` | Form → prediction bars → comorbidity step → recommendations + sources. |

**Chose → failed → changed (query-time):**
- **Retrieval: dense-only → hybrid (dense + BM25 + RRF).** Dense embeddings under-weight
  exact tokens; drug names (`empagliflozin`, `SGLT2`, `finerenone`) need lexical matching.
  Validated: hybrid > dense on recall/MRR, biggest gap on medication questions.
- **Queries: blind `"{stage} diabetes"` splice → stage-aware decomposition.** The naive
  version produced junk like `"No Diabetes diabetes"` and pulled treatment chunks for
  non-diabetics. Now prevention/screening queries for low-risk, treatment for Type 2.
- **Reranking: none → cross-encoder final filter.** Added `rerank_chunks` (ms-marco first,
  then **bge-reranker-base** — better MRR). Eval finding: reranking trades a little recall
  for top ranking, so it's **toggleable** (`FINAL_RERANK`). Runs on the *merged pool*, not
  per-query, as the last filter before the LLM.
- **Generator: local llama3.1 default; Claude opt-in.** A/B (below) showed Claude is clearly
  better, but `/recommend` sends patient data to the generator — so Claude means PHI leaves
  the machine. Kept llama3.1 as the **privacy-preserving default**.
- **Safety added incrementally** after observing failures: the model invented drug names when
  context lacked meds → deterministic drug redaction; ~13% of claims aren't fully grounded →
  runtime NLI tripwire; contradictory input → validation.

---

## Evaluation (how we know it works)  `rag/eval/`

| What | File | Result |
|---|---|---|
| Retrieval (embedding) | `retrieval_eval.py` | hybrid recall 0.88 / MRR 0.71 > dense 0.84 / 0.61 |
| Retrieval (LLM-judged) | `context_eval_llm.py` | recall 0.82 (cross-validates the embedding number) |
| Generation Tier-1 (no judge) | `generator_eval.py` | 0 fabricated doses; drug-grounding + redaction |
| Generation Tier-2 (Claude judge) | `judge_direct.py` | faithfulness 0.99 (Claude) vs 0.88 (llama) |
| Generator A/B | `compare_generators.py` | Claude wins: faith 0.99/rel 0.81 vs 0.88/0.51 |
| Visualization | `result.py` | `results.png` |

**Chose → failed → changed (eval):**
- **Local RAGAS judge → direct-Claude judge.** The 8B local judge ran ~245s/call (infeasible),
  the 3B was noisy, and RAGAS's `evaluate()` deadlocks on the async Anthropic client. So we
  compute faithfulness/relevancy with direct Claude calls (`judge_direct.py`).
- **Synthetic testset → hand-authored gold set.** RAGAS's `TestsetGenerator` crashed the
  Ollama runner; a curated 43-question set with `reference_contexts` replaced it.
- **RAGAS non-LLM context metrics → semantic matching.** String-distance scored relevant
  chunks ~0 (formatting differs); switched to embedding-similarity in `retrieval_eval.py`.

---

## Config knobs (env vars)

| Var | Default | Effect |
|---|---|---|
| `GENERATOR_MODEL` | `llama3.1` | `claude-sonnet-4-6` to generate via Anthropic (needs `ANTHROPIC_API_KEY`). |
| `FINAL_RERANK` | `1` | `0` to skip the cross-encoder rerank (RRF order instead). |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | swap the cross-encoder. |
| `GROUNDING_NLI` | `1` | `0` to skip the runtime NLI grounding check. |
| `JUDGE_MODEL` | `claude-sonnet-4-6` | judge model for the eval scripts. |

## Still open / would improve most
- A **patient-scenario testset** (vs. terse Q&A) — the biggest lever on trustworthy relevancy.
- **Table-aware extraction** — the one retrieval weak spot (diagnostic-criteria tables).
- A small **clinical-expert review** — the check no automated metric replaces.

---

## Function reference — purpose + input/output types

Recurring type: a **`chunk`** is a `dict` with keys `text` (str), `source_file` (str),
`heading` (str). After retrieval it also carries score keys (`rrf_score`, `dense_score`,
`bm25_score`, and `rerank_score` if reranked). `list[chunk]` means a Python list of those.

### `predict.py` — ML prediction
| Function | What it does | Input → Output |
|---|---|---|
| `preprocess_patient(raw)` | Raw patient values → the model's scaled feature row (one-hot + scaler) | `dict` → `pandas.DataFrame` (1×44) |
| `predict_patient(raw)` | Predict the diabetes stage and class probabilities | `dict` → `tuple[str, dict[str,float]]` |

### `rag/convert.py` — PDF → Markdown (build-time)
| Function | What it does | Input → Output |
|---|---|---|
| `main()` | Convert every `knowledge/pdfs/*.pdf` to `knowledge/md/*.md` (writes files) | `()` → `None` (side effect: files) |

### `rag/chunk.py` — semantic chunking (build-time)
| Function | What it does | Input → Output |
|---|---|---|
| `chunk_markdown(text, source_file)` | One Markdown doc → list of retrieval chunks | `(str, str)` → `list[chunk]` |
| `_iter_sections(text)` | Yield `(heading_trail, body)` per heading section, dropping noise lines | `str` → `generator[tuple[str, str]]` |
| `_segments(body)` | Split a section body into atomic segments (sentences; tables/lists kept whole) | `str` → `list[str]` |
| `_semantic_groups(segs)` | Group segments into chunk-sized strings at size/semantic breakpoints | `list[str]` → `list[str]` |
| `_hard_wrap(seg)` | Char-split an oversized single segment | `str` → `list[str]` |
| `_is_skippable(heading_trail)` / `_is_noise(line)` | Drop References/front-matter sections / running-header noise | `str` → `bool` |
| `_clean_heading(text)` | Strip Markdown emphasis/whitespace from a heading | `str` → `str` |

### `rag/embed.py` — embeddings
| Function | What it does | Input → Output |
|---|---|---|
| `get_model()` | Load/cache the sentence-transformers model | `()` → `SentenceTransformer` |
| `embed(texts)` | Embed strings → L2-normalized vectors | `list[str]` → `np.ndarray` (n×384, float32) |

### `rag/ingest.py` / `rag/store.py` — index build + vector store
| Function | What it does | Input → Output |
|---|---|---|
| `ingest.main()` | md → chunks → embeddings → save the index (writes files) | `()` → `None` |
| `save_index(embeddings, chunks)` | Persist `embeddings.npy` + `chunks.json` | `(np.ndarray, list[chunk])` → `None` |
| `load_index()` | Load the index into memory | `()` → `tuple[np.ndarray, list[chunk]]` |
| `search(query_vec, embeddings, chunks, k)` | Cosine top-k over the matrix | `(np.ndarray, np.ndarray, list[chunk], int)` → `list[chunk]` (+`score`) |

### `rag/retrieve.py` — hybrid retrieval + rerank
| Function | What it does | Input → Output |
|---|---|---|
| `tokenize(text)` | BM25 tokenizer (keeps `glp-1`, `sglt2`, digits) | `str` → `list[str]` |
| `rrf_fuse(rank_lists, rrf_k)` | Reciprocal Rank Fusion of ranked index lists | `list[list[int]]` → `list[tuple[int,float]]` |
| `get_reranker()` | Load/cache the cross-encoder reranker | `()` → `CrossEncoder` |
| `rerank_chunks(query, chunks, top_k)` | Cross-encoder rescore a pool → top_k reordered | `(str, list[chunk], int)` → `list[chunk]` (+`rerank_score`) |
| `HybridRetriever.__init__()` | Load index + build BM25 once | `()` → instance |
| `HybridRetriever.search(query, k, …)` | Dense + BM25 → RRF (→ optional per-query rerank) → top-k | `str` (+ ints/bools) → `list[chunk]` |

### `rag/generate.py` — query building, retrieval orchestration, generation
| Function | What it does | Input → Output |
|---|---|---|
| `summarize_patient(patient)` | Raw values → flagged clinical PROFILE text | `dict` → `str` |
| `_canonical(comorbidity)` | Map a comorbidity label to its canonical key (`stroke`→`ASCVD`) | `str` → `str | None` |
| `build_queries(stage, comorbidities)` | Form the stage-aware + per-comorbidity sub-queries | `(str, list[str])` → `list[tuple[str,str]]` (label, query) |
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

`generate_*` return `dict` keys: `recommendations` (str), `sources` (list[dict]), `warnings`
(list[str]), `redacted_drugs` (list[str]), `ungrounded_claims` (list[str]), `queries`,
`retrieved` (list[dict]).

### `rag/grounding.py` — runtime NLI grounding
| Function | What it does | Input → Output |
|---|---|---|
| `_claims(text)` | Split answer into checkable claim sentences | `str` → `list[str]` |
| `check_grounding(text, chunks, …)` | Per-claim NLI entailment vs context+profile | `(str, list[chunk])` → `dict{claims: list[dict], ungrounded: list[str]}` |

### `rag/validation.py` — guardrails
| Function | What it does | Input → Output |
|---|---|---|
| `validate_input(patient, comorbidities, stage)` | Input consistency + range checks | `(dict, list[str], str)` → `dict{errors: list[str], warnings: list[str]}` |
| `check_retrieval_coverage(chunks, stage, comorbidities)` | Comorbidities with no supporting chunk | `(list[chunk], str, list[str])` → `list[str]` |
| `check_output(text, n_sources)` | Headings/disclaimer/dose/citation checks | `(str, int)` → `list[str]` (warnings) |

### `db.py` — audit log
| Function | What it does | Input → Output |
|---|---|---|
| `init_db()` | Create the SQLite log table if absent | `()` → `None` |
| `log_recommendation(patient, stage, comorbidities, queries, retrieval)` | Insert one audit row | `(dict, str, list[str], list[dict], list[dict])` → `int` (row id) |

### `app.py` — FastAPI endpoints
| Function | What it does | Input → Output |
|---|---|---|
| `index(request)` | Render the patient form | `Request` → `HTMLResponse` |
| `predict(patient)` | `POST /predict` — stage + probabilities | `Patient` (pydantic) → `dict` |
| `recommend(req)` | `POST /recommend` — validate → generate → log | `RecommendRequest` (pydantic) → `dict` |
