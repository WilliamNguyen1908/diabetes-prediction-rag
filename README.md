# Diabetes Prediction + RAG Recommendations

Predicts a person's diabetes stage (No Diabetes / Pre-Diabetes / Type 2) with an
XGBoost model from a web form, then generates **personalized, guideline-grounded
health recommendations** with a RAG pipeline over 13 ADA clinical guideline PDFs
(hybrid retrieval + cross-encoder rerank + grounded LLM generation, with drug-name
redaction and NLI grounding guardrails).

## Run with Docker (recommended for trying it out)

Prerequisites: Docker with Compose.

```bash
docker compose up -d --build          # build the app image + start Ollama
docker compose exec ollama ollama pull llama3.1   # one-time, ~4.7 GB
```

Open **http://localhost:8000** — fill in the form, get a prediction, answer the
comorbidity questions, and request recommendations.

Notes:
- The first `/recommend` call loads the embedding/reranker/NLI models (~30 s);
  subsequent calls are just llama3.1 inference (~1–2 min on CPU).
- Everything runs locally — no patient data leaves the machine.
- To generate with Claude instead (better quality, but sends the patient profile
  to the Anthropic API): set `GENERATOR_MODEL=claude-sonnet-4-6` and
  `ANTHROPIC_API_KEY` in the app service environment (see `docker-compose.yml`).
- The audit log (`recommendation_log.db`) persists in the `app-data` volume.

**Building the image from a fresh checkout:** the image ships the prebuilt vector
index from `knowledge/index/`. If that directory is missing, build it first:

```bash
uv run python rag/convert.py && uv run python rag/ingest.py
```

## Run locally (development)

```bash
uv sync                                # Python 3.14 + all deps
uv run uvicorn app:app --reload        # needs Ollama running with llama3.1
```

## Repo map

| Path | What |
|---|---|
| `app.py` | FastAPI app: form UI, `POST /predict`, `POST /recommend` |
| `predict.py` | XGBoost inference (`diabetes_xgb.json` + `preprocess_bundle.joblib`) |
| `rag/` | RAG pipeline: convert → chunk → embed → ingest → retrieve → generate, plus guardrails (`validation.py`, `grounding.py`) |
| `rag/eval/` | Evaluation suite (retrieval metrics, LLM-judged generation, A/B) |
| `db.py` | SQLite audit log of every recommendation request |
| `CLAUDE.md` | Full architecture documentation: pipeline map, design history, function reference |
