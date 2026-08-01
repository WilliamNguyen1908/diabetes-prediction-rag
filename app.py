"""FastAPI app: serves the patient input form, /predict, and /recommend (RAG).

Run:  uv run uvicorn app:app --reload
Then open http://127.0.0.1:8000
"""
import os
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import db
import predict as P

# rag/ modules use flat imports (from retrieve import ...) so they can also run as
# scripts; put rag/ on sys.path so the app can import them the same way.
sys.path.insert(0, str(Path(__file__).parent / "rag"))

from validation import validate_input  # noqa: E402

app = FastAPI(title="Diabetes Stage Prediction")
templates = Jinja2Templates(directory="templates")


# --- rate limiting for /recommend (the only endpoint that calls a paid LLM) -----
# In-memory, single-process — right-sized for a single-container demo (e.g. a
# Hugging Face Space). Two independent limits, both tunable via env:
#   * per-IP  hourly  — stops one visitor from monopolising the demo
#   * global  daily   — a hard cost ceiling so a public URL can't run up the bill
_PER_IP_PER_HOUR = int(os.environ.get("RECOMMEND_PER_IP_PER_HOUR", "5"))
_GLOBAL_PER_DAY = int(os.environ.get("RECOMMEND_GLOBAL_PER_DAY", "20"))
_ip_hits: dict[str, deque] = defaultdict(deque)
_global_hits: deque = deque()
_rl_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    """Real client IP — behind HF/other proxies it's first in X-Forwarded-For."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request):
    """Raise 429 if the per-IP hourly or global daily quota is exceeded."""
    now = time.time()
    ip = _client_ip(request)
    with _rl_lock:
        while _global_hits and _global_hits[0] < now - 86_400:
            _global_hits.popleft()
        hits = _ip_hits[ip]
        while hits and hits[0] < now - 3_600:
            hits.popleft()
        if not hits:                       # keep the map from growing unbounded
            _ip_hits.pop(ip, None)
            hits = _ip_hits[ip]
        if len(_global_hits) >= _GLOBAL_PER_DAY:
            raise HTTPException(status_code=429, detail={
                "message": "This demo has reached its daily usage cap. Please try again tomorrow."})
        if len(hits) >= _PER_IP_PER_HOUR:
            raise HTTPException(status_code=429, detail={
                "message": f"Rate limit reached ({_PER_IP_PER_HOUR}/hour). Please wait a bit and retry."})
        hits.append(now)
        _global_hits.append(now)


# --- input schema: validated against the model's own choices/fields ------------
class Patient(BaseModel):
    # numeric fields (19)
    age: float
    bmi: float
    waist_to_hip_ratio: float
    physical_activity_minutes_per_week: float
    diet_score: float
    alcohol_consumption_per_week: float
    sleep_hours_per_day: float
    screen_time_hours_per_day: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    cholesterol_total: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    triglycerides: float
    glucose_fasting: float
    glucose_postprandial: float
    insulin_level: float
    hba1c: float
    # binary fields (3) — 0 / 1
    family_history_diabetes: int
    hypertension_history: int
    cardiovascular_history: int
    # categorical fields (6)
    gender: str
    ethnicity: str
    education_level: str
    income_level: str
    employment_status: str
    smoking_status: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Render the input form, generated from the model's bundle metadata."""
    numeric = [{"name": k, "default": v} for k, v in P.NUM_DEFAULTS.items()]
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "numeric": numeric,
            "binary": P.BINARY,
            "choices": P.CHOICES,
        },
    )


@app.post("/predict")
def predict(patient: Patient):
    """Return the predicted diabetes stage and class probabilities."""
    label, proba = P.predict_patient(patient.model_dump())
    return {"stage": label, "probabilities": proba}


class RecommendRequest(Patient):
    # Comorbidity answers gathered after the prediction (free-form list of terms,
    # e.g. ["heart failure", "chronic kidney disease", "stroke"]). Folded into the
    # RAG retrieval query where BM25 pulls the matching guideline chunks.
    comorbidities: list[str] = Field(default_factory=list)


@app.post("/recommend")
def recommend(req: RecommendRequest, request: Request):
    """Predict the stage, then generate RAG-grounded recommendations (LLM)."""
    _enforce_rate_limit(request)  # gate the paid LLM call before any work
    from generate import generate_recommendations  # lazy: loads embedder/LLM on first call

    patient = req.model_dump()
    comorbidities = patient.pop("comorbidities", [])
    stage, proba = P.predict_patient(patient)

    # Guardrail A: reject contradictory / impossible input BEFORE the expensive
    # retrieval + generation (e.g. stroke=Yes with cardiovascular history=No).
    checks = validate_input(patient, comorbidities, stage)
    if checks["errors"]:
        raise HTTPException(status_code=400, detail={
            "message": "Input failed consistency checks — please reconcile and resubmit.",
            "errors": checks["errors"], "warnings": checks["warnings"], "stage": stage,
        })

    # Pass the full patient dict so recommendations are personalized to their own
    # values (a flagged clinical profile is built inside generate_recommendations).
    result = generate_recommendations(stage=stage, comorbidities=comorbidities, patient=patient)

    # Audit log: patient profile + formed sub-queries + retrieved chunks (SQLite).
    try:
        db.log_recommendation(patient, stage, comorbidities,
                              result.get("queries", []), result.get("retrieved", []))
    except Exception as e:  # never let logging break a response
        print(f"[log] failed to write audit record: {e}")

    return {
        "stage": stage,
        "probabilities": proba,
        "recommendations": result["recommendations"],
        "sources": result["sources"],
        "warnings": checks["warnings"] + result.get("warnings", []),
    }
