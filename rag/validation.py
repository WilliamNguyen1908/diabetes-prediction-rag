"""Deterministic guardrails for the recommendation pipeline.

Implements the automatable checks from the guardrail spec:
  - validate_input        (Section A): input consistency, ranges, stage-vs-labs.
  - check_retrieval_coverage (Section B): every Yes comorbidity + the stage are
    represented in the retrieved context.
  - check_output          (Sections C/F, structural): doses, headings, disclaimer,
    citation mapping. (Drug-name grounding is enforced separately by
    generate.apply_drug_safety.)

Semantic checks (does the prose actually address each abnormal value, comorbidity
ordering, "borderline vs high" wording) are NOT here — they need an LLM judge; the
generation prompt is what enforces them.

`errors` are hard contradictions/impossibilities -> reject before generating.
`warnings` are soft flags -> surface to the user but proceed.
"""
import re

# Physiologically plausible ranges; a value outside these is impossible -> reject.
_RANGES = {
    "age": (0, 120), "bmi": (10, 70), "hba1c": (3, 20),
    "glucose_fasting": (20, 700), "glucose_postprandial": (20, 900),
    "systolic_bp": (50, 300), "diastolic_bp": (30, 200), "heart_rate": (20, 250),
    "cholesterol_total": (50, 800), "hdl_cholesterol": (5, 200),
    "ldl_cholesterol": (10, 500), "triglycerides": (20, 5000),
    "insulin_level": (0, 500), "waist_to_hip_ratio": (0.4, 2.0),
}

# Comorbidity -> context terms that count as "this branch was covered by retrieval".
_COMORBIDITY_TERMS = {
    "heart failure": ["heart failure", "hfref", "hfpef", "ejection fraction"],
    "chronic kidney disease": ["kidney", "ckd", "nephropathy", "albuminuria", "egfr"],
    "atherosclerotic cardiovascular disease": ["cardiovascular", "ascvd", "atherosclerotic", "coronary"],
    "stroke": ["stroke", "cerebrovascular", "cardiovascular", "ascvd"],
    "hypertension": ["hypertension", "blood pressure"],
    "obesity": ["obesity", "overweight", "weight loss", "bariatric"],
}

_REQUIRED_HEADINGS = ["diet & nutrition", "physical activity", "monitoring",
                      "medication considerations", "when to seek care"]
_DISCLAIMER = "not a substitute for professional medical advice"
_DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|units?)\b(?!\s*/)", re.I)


def _mentions(comorbidities, *keys):
    low = " ".join(comorbidities).lower()
    return any(k in low for k in keys)


def validate_input(patient: dict, comorbidities=None, stage: str = None) -> dict:
    """Section A. Returns {'errors': [...], 'warnings': [...]}."""
    comorbidities = comorbidities or []
    errors, warnings = [], []

    # Required fields
    for f in ("age", "bmi"):
        if patient.get(f) is None:
            errors.append(f"Required field '{f}' is missing.")
    if patient.get("hba1c") is None and patient.get("glucose_fasting") is None:
        errors.append("At least one glycemic value (HbA1c or fasting glucose) is required.")

    # Physiological ranges — impossible values are rejected
    for f, (lo, hi) in _RANGES.items():
        v = patient.get(f)
        if v is not None and not (lo <= v <= hi):
            errors.append(f"{f} = {v} is outside the plausible range {lo}–{hi}.")

    # Input contradictions
    stroke_or_ascvd = _mentions(comorbidities, "stroke", "ascvd", "atherosclerotic", "cardiovascular")
    if stroke_or_ascvd and patient.get("cardiovascular_history") == 0:
        errors.append("Stroke/ASCVD is marked Yes but cardiovascular history = No — a stroke/ASCVD "
                      "IS cardiovascular history. Set cardiovascular history to Yes, or remove the comorbidity.")

    bp_high = (patient.get("systolic_bp") or 0) >= 140 or (patient.get("diastolic_bp") or 0) >= 90
    if (_mentions(comorbidities, "hypertension") or bp_high) and patient.get("hypertension_history") == 0:
        warnings.append("Hypertension indicated (flag or BP ≥140/90) but hypertension history = No — please confirm.")

    bmi = patient.get("bmi")
    if _mentions(comorbidities, "obesity") and bmi is not None and bmi < 25:
        warnings.append(f"Obesity marked Yes but BMI = {bmi:.1f} (<25) — the flag contradicts the measurement.")

    # Stage vs. labs — reconcile the classifier against the entered values
    a1c, fg = patient.get("hba1c"), patient.get("glucose_fasting")
    if stage == "No Diabetes" and ((a1c is not None and a1c >= 6.5) or (fg is not None and fg >= 126)):
        warnings.append(f"Predicted 'No Diabetes' but labs are in the diabetes range "
                        f"(HbA1c {a1c}, fasting {fg}) — reconcile against the classifier.")
    if stage == "Type 2" and a1c is not None and a1c < 5.7 and (fg is None or fg < 100):
        warnings.append(f"Predicted 'Type 2' but HbA1c {a1c}% and fasting glucose are normal — reconcile.")

    # Low-prior-but-not-impossible combinations -> confirm, don't silently use
    age = patient.get("age")
    if age is not None and age < 30 and stroke_or_ascvd:
        warnings.append(f"Age {age:.0f} with stroke/ASCVD is uncommon — please confirm before a "
                        "secondary-prevention plan.")

    return {"errors": errors, "warnings": warnings}


def check_retrieval_coverage(chunks, stage: str, comorbidities=None) -> list:
    """Section B. Return the comorbidities that have NO supporting chunk in the context."""
    comorbidities = comorbidities or []
    ctx = " ".join(c["text"] for c in chunks).lower()
    missing = []
    for c in comorbidities:
        terms = _COMORBIDITY_TERMS.get(c.lower(), [c.lower()])
        if not any(t in ctx for t in terms):
            missing.append(c)
    return missing


def check_output(text: str, n_sources: int) -> list:
    """Sections C/F (structural). Return warnings about the generated text."""
    warnings = []
    low = text.lower()
    for h in _REQUIRED_HEADINGS:
        if h not in low:
            warnings.append(f"Output is missing the '{h}' section.")
    if _DISCLAIMER not in low:
        warnings.append("Output is missing the safety disclaimer.")
    if _DOSE_RE.search(text):
        warnings.append("Output appears to contain a medication dose (prompt forbids dosing).")
    for m in set(re.findall(r"\[(\d+)\]", text)):
        if not (1 <= int(m) <= n_sources):
            warnings.append(f"Citation [{m}] does not map to a retrieved source (1–{n_sources}).")
    return warnings
