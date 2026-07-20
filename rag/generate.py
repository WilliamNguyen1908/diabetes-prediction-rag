"""Recommendation generation: retrieve grounded context, then generate with a local LLM.

Flow: predicted stage (+ optional comorbidity answers) -> two focused retrieval
queries (lifestyle + medication) -> merged guideline context -> llama3.1 via
Ollama produces recommendations grounded ONLY in that context.

The medication queries fold the patient's comorbidities in as tokens, which is
where hybrid retrieval's BM25 half pulls the right guideline chunks (e.g. SGLT2i
/ finerenone for heart failure / CKD).

Usage:
    from generate import generate_recommendations
    out = generate_recommendations(stage="Type 2", comorbidities=["heart failure"])
    print(out["recommendations"]); print(out["sources"])
"""
import os
import re
from functools import lru_cache

import ollama

from retrieve import HybridRetriever, rerank_chunks
from validation import check_output, check_retrieval_coverage

# Final-pool cross-encoder rerank before the LLM (on by default; FINAL_RERANK=0 to disable).
# Reranks the merged candidate pool against a combined patient query, then takes the top-K.
# Tradeoff (from eval): improves top ranking, can trade a little recall — so it's toggleable.
FINAL_RERANK = os.environ.get("FINAL_RERANK", "1") == "1"

# Generator model. Default: local llama3.1 via Ollama. Set GENERATOR_MODEL to a Claude
# model id (e.g. claude-sonnet-4-6) to generate via the Anthropic API instead — needs
# ANTHROPIC_API_KEY. Everything else (retrieval, safety filter, guardrails) is unchanged.
MODEL = os.environ.get("GENERATOR_MODEL", "llama3.1")
_IS_CLAUDE = MODEL.startswith("claude")
PER_QUERY_K = 4          # chunks retrieved per sub-query
MAX_CONTEXT_CHUNKS = 12  # cap merged context fed to the LLM (room for lifestyle + several drug classes)


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic
    return anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment


def _chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
    """Single chat completion, routed to Claude (Anthropic API) or Ollama by MODEL."""
    if _IS_CLAUDE:
        resp = _anthropic_client().messages.create(
            model=MODEL, system=system or "", max_tokens=max_tokens,
            temperature=temperature, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
    return ollama.chat(model=MODEL, messages=messages, options={"temperature": temperature})["message"]["content"]

# --- Drug-grounding safety net -------------------------------------------------
# Deterministic backstop on top of the prompt: any specific medication the LLM names
# that is NOT supported by the retrieved context (neither the agent nor its drug class
# appears) is redacted before the text reaches the user. A hallucinated drug name is the
# highest-risk failure for this system, so we never rely on the model alone.
DRUG_AGENTS = [
    # Glucose-lowering
    "metformin",
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "bexagliflozin",
    "semaglutide", "liraglutide", "dulaglutide", "exenatide", "lixisenatide", "tirzepatide",
    "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
    "glipizide", "glimepiride", "glyburide", "gliclazide",
    "pioglitazone", "rosiglitazone",
    "repaglinide", "nateglinide", "acarbose", "miglitol",
    "glargine", "lispro", "aspart", "detemir", "degludec",
    # Kidney
    "finerenone",
    # Lipid-lowering
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin",
    "pitavastatin", "fluvastatin", "ezetimibe",
    # Blood pressure (ACE inhibitors / ARBs / others)
    "lisinopril", "ramipril", "enalapril", "benazepril", "perindopril", "captopril",
    "losartan", "valsartan", "olmesartan", "telmisartan", "irbesartan", "candesartan",
    "amlodipine", "hydrochlorothiazide", "chlorthalidone",
    # Antiplatelet
    "aspirin", "clopidogrel",
]
_ACE_INHIBITORS = {"lisinopril", "ramipril", "enalapril", "benazepril", "perindopril", "captopril"}
_NEUTRAL = "a medication your clinician can determine"


def drug_class_token(drug: str) -> str:
    """The class token to look for in context when the exact agent name is absent.
    Tokens are chosen to be specific enough to avoid false substring matches
    (e.g. 'statin', not 'arb' which would match 'carbohydrate')."""
    if drug.endswith("flozin"):
        return "sglt"
    if drug.endswith("gliptin"):
        return "dpp"
    if drug in ("semaglutide", "liraglutide", "dulaglutide", "exenatide", "lixisenatide", "tirzepatide"):
        return "glp"
    if drug in ("glipizide", "glimepiride", "glyburide", "gliclazide"):
        return "sulfonylurea"
    if drug in ("pioglitazone", "rosiglitazone"):
        return "glitazone"
    if drug.endswith("statin"):
        return "statin"
    if drug.endswith("sartan"):
        return "angiotensin receptor"     # ARBs
    if drug in _ACE_INHIBITORS:
        return "ace inhibitor"
    # metformin, finerenone, insulins, ezetimibe, CCBs, diuretics, antiplatelets:
    # their own name is the grounding token.
    return drug


def find_drug_agents(text: str):
    low = text.lower()
    return sorted({d for d in DRUG_AGENTS if re.search(rf"\b{re.escape(d)}\b", low)})


def ungrounded_agents(text: str, context_lower: str):
    """Agents named in `text` that are supported by neither their name nor class in context."""
    return [d for d in find_drug_agents(text)
            if d not in context_lower and drug_class_token(d) not in context_lower]


def apply_drug_safety(text: str, chunks):
    """Redact ungrounded drug names from the generated text. Returns (clean_text, redacted)."""
    context_lower = " ".join(c["text"] for c in chunks).lower()
    redacted = ungrounded_agents(text, context_lower)
    out = text
    for d in redacted:
        # Drop an enclosing example parenthetical, e.g. "(such as empagliflozin)".
        out = re.sub(rf"\s*\((?:such as|e\.g\.,?|like|including)[^)]*\b{re.escape(d)}\b[^)]*\)", "", out, flags=re.I)
        # Neutralize any remaining bare mentions.
        out = re.sub(rf"\b{re.escape(d)}\b", _NEUTRAL, out, flags=re.I)
    return out, redacted

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for diabetes care. "
    "Use ONLY the numbered guideline excerpts provided as CONTEXT. "
    "Perform a careful, evidence-based synthesis of the CONTEXT and the PATIENT PROFILE. "
    "Create recommendations for patient when they should do screening even though they do not have diabetes yet but recommendation should based on the patient current information and the context"
    "Do not invent facts, drug names, doses, or targets that are not in the context; "
    "if the context is insufficient for something, say so plainly. "
    "Cite the excerpts you rely on inline as [1], [2], etc. "
    "PERSONALIZE every recommendation to the PATIENT PROFILE: explicitly reference the "
    "patient's OWN values and the flagged abnormalities, e.g. 'Because your LDL cholesterol "
    "is 160 mg/dL (high), ...' or 'Given your age (62) and BMI of 33 (obese), ...'. Lead each "
    "section with the specific measurements that motivate the advice, and prioritize the "
    "values flagged as abnormal. Do not give generic advice that ignores their numbers. "
    "For Medication Considerations, be SPECIFIC: name the relevant medication CLASSES and "
    "representative example agents that appear in the context — e.g. 'SGLT2 inhibitors "
    "(such as empagliflozin)' or 'GLP-1 receptor agonists (such as semaglutide)' — and for "
    "each, briefly say why it fits THIS patient's comorbidities and predicted stage (e.g. "
    "cardiovascular, heart failure, kidney, or stroke-risk benefit). Only name classes or "
    "drugs that are supported by the context; do NOT invent agents, and do NOT give specific "
    "doses or titration — note that the final drug choice and dosing are a clinician's decision. "
    "CRITICAL: never introduce a drug name (e.g. metformin, empagliflozin, semaglutide) from "
    "your own knowledge if it does not literally appear in the CONTEXT. This applies to ANY "
    "medication class the context may cover — glucose-lowering, cardiovascular, kidney "
    "(e.g. finerenone), lipid-lowering (e.g. statins), or blood-pressure agents. If the "
    "CONTEXT does not discuss any medications at all, the Medication Considerations section "
    "must say 'The retrieved guidelines do not provide specific medication guidance for this "
    "case' and name NO drugs. "
    "Do NOT contradict yourself: either the context supports specific medications (name them "
    "and give the rationale) OR it does not (say so and stop) — never state that guidance is "
    "unavailable and then go on to recommend medications anyway. "
    "Be concise and organize the answer under these headings: "
    "Diet & Nutrition, Physical Activity & Lifestyle, Monitoring, "
    "Medication Considerations, When to Seek Care. "
    "End with one line: 'This is educational information, not a substitute for professional medical advice.'"
)


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever()


def summarize_patient(patient: dict) -> str:
    """Turn the raw form input into a clinically-flagged profile the LLM can cite.

    Each line is 'Label: value (interpretation)', and abnormal values are collected
    into a 'Notable abnormal findings' line so the model leads with what matters.
    Thresholds follow common clinical reference ranges (ADA / ATP III / ACC-AHA).
    """
    p = patient
    female = str(p.get("gender", "")).lower().startswith("f")
    lines, notable = [], []

    def add(label, text, abnormal=None):
        lines.append(f"- {label}: {text}")
        if abnormal:
            notable.append(abnormal)

    add("Age", f"{p['age']:.0f} years")

    bmi = p["bmi"]
    b = ("obese" if bmi >= 30 else "overweight" if bmi >= 25 else
         "normal" if bmi >= 18.5 else "underweight")
    add("BMI", f"{bmi:.1f} ({b})", f"BMI {bmi:.1f} ({b})" if b in ("obese", "overweight", "underweight") else None)

    whr = p["waist_to_hip_ratio"]
    whr_hi = whr > (0.85 if female else 0.90)
    add("Waist-to-hip ratio", f"{whr:.2f}" + (" (elevated)" if whr_hi else ""),
        f"elevated waist-to-hip ratio ({whr:.2f})" if whr_hi else None)

    a1c = p["hba1c"]
    a1c_s = ("high — diabetes range" if a1c >= 6.5 else "elevated — prediabetes range" if a1c >= 5.7 else "normal")
    add("HbA1c", f"{a1c:.1f}% ({a1c_s})", f"HbA1c {a1c:.1f}% ({a1c_s})" if a1c >= 5.7 else None)

    fg = p["glucose_fasting"]
    fg_s = ("high — diabetes range" if fg >= 126 else "impaired" if fg >= 100 else "normal")
    add("Fasting glucose", f"{fg:.0f} mg/dL ({fg_s})", f"fasting glucose {fg:.0f} ({fg_s})" if fg >= 100 else None)

    pg = p["glucose_postprandial"]
    pg_s = ("high — diabetes range" if pg >= 200 else "impaired" if pg >= 140 else "normal")
    add("Post-meal glucose", f"{pg:.0f} mg/dL ({pg_s})", f"post-meal glucose {pg:.0f} ({pg_s})" if pg >= 140 else None)

    tc = p["cholesterol_total"]
    tc_s = ("high" if tc >= 240 else "borderline high" if tc >= 200 else "desirable")
    add("Total cholesterol", f"{tc:.0f} mg/dL ({tc_s})", f"total cholesterol {tc:.0f} ({tc_s})" if tc >= 200 else None)

    hdl = p["hdl_cholesterol"]
    hdl_lo = hdl < (50 if female else 40)
    add("HDL cholesterol", f"{hdl:.0f} mg/dL" + (" (low)" if hdl_lo else ""),
        f"low HDL ({hdl:.0f})" if hdl_lo else None)

    ldl = p["ldl_cholesterol"]
    ldl_s = ("very high" if ldl >= 190 else "high" if ldl >= 160 else "borderline high" if ldl >= 130 else
             "near optimal" if ldl >= 100 else "optimal")
    add("LDL cholesterol", f"{ldl:.0f} mg/dL ({ldl_s})", f"LDL {ldl:.0f} ({ldl_s})" if ldl >= 130 else None)

    tg = p["triglycerides"]
    tg_s = ("very high" if tg >= 500 else "high" if tg >= 200 else "borderline high" if tg >= 150 else "normal")
    add("Triglycerides", f"{tg:.0f} mg/dL ({tg_s})", f"triglycerides {tg:.0f} ({tg_s})" if tg >= 150 else None)

    sbp, dbp = p["systolic_bp"], p["diastolic_bp"]
    if sbp >= 140 or dbp >= 90:
        bp_s = "stage 2 hypertension"
    elif sbp >= 130 or dbp >= 80:
        bp_s = "stage 1 hypertension"
    elif sbp >= 120:
        bp_s = "elevated"
    else:
        bp_s = "normal"
    add("Blood pressure", f"{sbp:.0f}/{dbp:.0f} mmHg ({bp_s})",
        f"blood pressure {sbp:.0f}/{dbp:.0f} ({bp_s})" if "hypertension" in bp_s or bp_s == "elevated" else None)

    hr = p["heart_rate"]
    hr_s = ("high" if hr > 100 else "low" if hr < 60 else "normal")
    add("Resting heart rate", f"{hr:.0f} bpm ({hr_s})", f"heart rate {hr:.0f} ({hr_s})" if hr_s != "normal" else None)

    pa = p["physical_activity_minutes_per_week"]
    add("Physical activity", f"{pa:.0f} min/week" + (" (below recommended 150)" if pa < 150 else ""),
        f"low physical activity ({pa:.0f} min/week)" if pa < 150 else None)

    sl = p["sleep_hours_per_day"]
    add("Sleep", f"{sl:.1f} h/night" + (" (outside 7-9)" if not (7 <= sl <= 9) else ""),
        f"suboptimal sleep ({sl:.1f} h)" if not (7 <= sl <= 9) else None)

    ds = p["diet_score"]
    add("Diet score", f"{ds:.0f}/10" + (" (poor)" if ds < 5 else ""),
        f"poor diet score ({ds:.0f}/10)" if ds < 5 else None)

    alc = p["alcohol_consumption_per_week"]
    alc_hi = alc > (7 if female else 14)
    add("Alcohol", f"{alc:.0f} drinks/week" + (" (above recommended)" if alc_hi else ""),
        f"high alcohol intake ({alc:.0f}/week)" if alc_hi else None)

    ins = p["insulin_level"]
    add("Fasting insulin", f"{ins:.0f} µIU/mL" + (" (elevated — possible insulin resistance)" if ins > 25 else ""),
        f"elevated fasting insulin ({ins:.0f})" if ins > 25 else None)

    profile = "PATIENT PROFILE (values the patient entered):\n" + "\n".join(lines)
    if notable:
        profile += "\n\nNotable abnormal findings: " + "; ".join(notable) + "."
    return profile


# Stage-appropriate retrieval phrasing (keys match the model's stage labels). Blindly
# splicing "{stage} diabetes" produced junk like "No Diabetes diabetes" and pulled
# glucose-lowering-treatment chunks for non-diabetics; these focus each stage correctly:
# prevention/screening for non-diabetic & prediabetic, management + treatment for type 2.
_STAGE_QUERIES = {
    "No Diabetes": [
        "diet, nutrition, and physical activity to prevent type 2 diabetes and reduce cardiometabolic risk factors",
        "screening for type 2 diabetes and prediabetes in asymptomatic adults, age to begin screening and rescreening interval",
        "healthy eating patterns, weight management, and exercise to maintain normoglycemia and lower diabetes risk",
        "risk factors for type 2 diabetes and when to test overweight or obese adults regardless of age",
    ],
    "Pre-Diabetes": [
        "prediabetes lifestyle intervention: reduced-calorie diet, 5–7% weight loss goal, and 150 minutes weekly physical activity to prevent type 2 diabetes",
        "evidence-based eating patterns such as Mediterranean or low-carbohydrate for adults with prediabetes",
        "metformin and pharmacologic options to prevent or delay type 2 diabetes in high-risk adults with prediabetes",
        "diabetes prevention program structure, weight-loss and activity goals, and effectiveness for delaying progression",
        "monitoring and annual testing for progression from prediabetes to diabetes; sleep and behavioral factors",
    ],
    "Type 2": [
    # Diet / nutrition — doc5 (§5)
    "type 2 diabetes medical nutrition therapy and eating patterns: Mediterranean, DASH, "
    "low-carbohydrate, carbohydrate counting, fiber, sodium reduction, and weight loss",

    # Physical activity — doc5 (§5)
    "physical activity for type 2 diabetes: 150 minutes per week of moderate-to-vigorous "
    "aerobic exercise, 2-3 resistance training sessions, and breaking up sedentary time",

    # Behavioral / self-management — doc5 (§5)
    "diabetes self-management education and support, sleep, and behavioral strategies for "
    "type 2 diabetes glycemic control",

    # Drug selection (foundational) — doc9 (§9)
    "glucose-lowering medication selection for type 2 diabetes: metformin, early combination "
    "therapy, and choosing agents by comorbidity, weight, and hypoglycemia risk",

    # Cardiorenal-benefit agents — doc9 / doc10 / doc11
    "SGLT2 inhibitors and GLP-1 receptor agonists with cardiovascular and kidney benefit for "
    "type 2 diabetes with ASCVD, heart failure, or chronic kidney disease",

    # Insulin / injectable intensification — doc9 (§9)
    "insulin initiation and intensification to injectable therapy in type 2 diabetes: basal "
    "insulin and GLP-1 receptor agonist combination",

    # Weight-loss pharmacotherapy + surgery — doc8 (§8)
    "weight management for type 2 diabetes with overweight or obesity: GLP-1 and dual "
    "GIP/GLP-1 receptor agonists (semaglutide, tirzepatide), obesity pharmacotherapy, and "
    "metabolic surgery",

    # Glycemic targets & monitoring — doc6 (§6)
    "individualized A1C and time-in-range glycemic targets, glucose monitoring, and "
    "hypoglycemia prevention in type 2 diabetes",

    # Cardiovascular risk factors (BP + lipids) — doc10 (§10)   << the main addition
    "blood pressure and lipid management in type 2 diabetes: ACE inhibitor or ARB, blood "
    "pressure target of 130/80, high-intensity statin, ezetimibe, PCSK9 inhibitor, and "
    "icosapent ethyl for high triglycerides",
],
}


# Only the six comorbidities in your form.
_COMORBIDITY_QUERIES = {
    "ASCVD":        ("diabetes with established atherosclerotic cardiovascular disease, prior stroke or "
                     "myocardial infarction: GLP-1 receptor agonists and SGLT2 inhibitors with proven "
                     "cardiovascular benefit, high-intensity statin, aspirin for secondary prevention, "
                     "ACE inhibitor or ARB"),
    "HeartFailure": ("type 2 diabetes with heart failure: SGLT2 inhibitor to reduce heart failure "
                     "hospitalizations regardless of ejection fraction; dual GIP/GLP-1 receptor agonist for HFpEF"),
    "CKD":          ("type 2 diabetes and chronic kidney disease with albuminuria: ACE inhibitor or ARB, "
                     "SGLT2 inhibitor, GLP-1 receptor agonist, and nonsteroidal mineralocorticoid receptor "
                     "antagonist finerenone"),
    "Hypertension": ("diabetes and hypertension blood pressure targets and first-line agents: ACE inhibitor or "
                     "ARB, thiazide-like diuretic, calcium channel blocker, combination therapy"),
    "Obesity":      ("diabetes with overweight or obesity: GLP-1 and dual GIP/GLP-1 receptor agonists "
                     "(semaglutide, tirzepatide) for weight loss, obesity pharmacotherapy, and metabolic surgery"),
}

# Exact toggle labels (+ a few variants) -> canonical key.
_COMORBIDITY_ALIASES = {
    "heart failure": "HeartFailure", "hf": "HeartFailure",
    "chronic kidney disease": "CKD", "ckd": "CKD", "kidney disease": "CKD",
    "cardiovascular disease (ascvd)": "ASCVD", "cardiovascular disease": "ASCVD",
    "ascvd": "ASCVD", "cvd": "ASCVD",
    "stroke": "ASCVD",              # stroke is an ASCVD event -> same drug guidance
    "hypertension": "Hypertension", "high blood pressure": "Hypertension", "htn": "Hypertension",
    "obesity": "Obesity", "overweight": "Obesity",
}

_GENERIC_COMORBIDITY_QUERY = (
    "glucose-lowering medication class and agents for diabetes with {c}: "
    "SGLT2 inhibitors, GLP-1 receptor agonists, metformin, pioglitazone, finerenone"
)


def _canonical(comorbidity: str) -> str | None:
    key = comorbidity.strip().lower()
    if key in _COMORBIDITY_ALIASES:
        return _COMORBIDITY_ALIASES[key]
    # tolerate the "(ASCVD)" style suffix or extra spacing
    import re
    key = re.sub(r"\(.*?\)", "", key)          # drop parenthetical
    key = re.sub(r"\s+", " ", key).strip()
    return _COMORBIDITY_ALIASES.get(key)


def build_queries(stage: str, comorbidities=None):
    """Stage-appropriate retrieval sub-queries plus ONE tailored medication query per
    comorbidity, so the correct drug-specific guideline chunks (SGLT2i for heart failure,
    SGLT2i/finerenone/ACEi-ARB for CKD, GLP-1/SGLT2i + statin/aspirin for prior stroke or
    ASCVD, ACEi/ARB for hypertension, GLP-1/tirzepatide for obesity) surface as high-value
    BM25 tokens instead of a generic one-size-fits-all list."""
    comorbidities = comorbidities or []
    # Each entry is (source_label, query) so callers can see how the query was formed.
    labeled = [(f"stage:{stage}", q) for q in _STAGE_QUERIES.get(stage, _STAGE_QUERIES["Type 2"])]
    for c in comorbidities:
        canon = _canonical(c)
        if canon:
            labeled.append((f"comorbidity:{c}→{canon}", _COMORBIDITY_QUERIES[canon]))
        else:
            labeled.append((f"comorbidity:{c}→generic", _GENERIC_COMORBIDITY_QUERY.format(c=c)))
    # de-dup by query text while preserving order (Stroke + ASCVD -> single ASCVD query)
    seen, out = set(), []
    for label, q in labeled:
        if q not in seen:
            seen.add(q); out.append((label, q))
    return out


def _rewrite_query(topic: str, stage: str):
    """CRAG-style query rewrite: ask the LLM for a better retrieval query for an
    uncovered topic. Returns a single query string, or None on failure."""
    try:
        text = _chat(
            system=None,
            user=(f"Write ONE concise search query (max 18 words) to find clinical guideline "
                  f"text about managing {topic} in a patient with {stage}. "
                  f"Return only the query text — no quotes, no preamble."),
            temperature=0.0, max_tokens=64,
        )
        q = text.strip().strip('"').splitlines()[0]
        return q or None
    except Exception:
        return None


def _combined_query(stage: str, comorbidities=None):
    """A single representative query for the final-pool rerank (the cross-encoder needs one)."""
    q = (f"{stage} diabetes management recommendations: diet, physical activity, "
         f"glucose-lowering medication, monitoring")
    if comorbidities:
        q += " with " + ", ".join(comorbidities)
    return q


def retrieve_context(stage: str, comorbidities=None):
    """Retrieve + merge sub-query hits (deduped), run one round of CORRECTIVE retrieval for
    uncovered comorbidities, then a FINAL cross-encoder rerank of the merged pool against a
    combined patient query (RRF order if FINAL_RERANK is off) before feeding the LLM."""
    r = _retriever()
    seen, merged = set(), []

    def add_hits(query):
        for hit in r.search(query, k=PER_QUERY_K):
            key = (hit["source_file"], hit["text"][:80])
            if key not in seen:
                seen.add(key)
                merged.append(hit)

    labeled_queries = build_queries(stage, comorbidities)
    print(f"\n[retrieval] stage={stage!r} comorbidities={comorbidities or []}")
    print(f"[retrieval] formed {len(labeled_queries)} sub-quer(ies) "
          f"({len(_STAGE_QUERIES.get(stage, _STAGE_QUERIES['Type 2']))} stage-base + one per comorbidity, deduped):")
    for i, (label, q) in enumerate(labeled_queries, 1):
        print(f"[retrieval]   query {i} [{label}]: {q}")
        add_hits(q)

    combined = _combined_query(stage, comorbidities)

    def select_top():
        if FINAL_RERANK:
            return rerank_chunks(combined, merged, MAX_CONTEXT_CHUNKS)
        merged.sort(key=lambda h: -h["rrf_score"])
        return merged[:MAX_CONTEXT_CHUNKS]

    top = select_top()

    # --- Corrective retrieval (self-reflection): rewrite + retry uncovered branches ---
    missing = check_retrieval_coverage(top, stage, comorbidities)
    if missing:
        print(f"[retrieval] coverage gap for {missing} — rewriting queries and retrying")
        for c in missing:
            new_q = _rewrite_query(c, stage)
            if new_q:
                print(f"[retrieval]   rewrite for '{c}': {new_q}")
                add_hits(new_q)
        top = select_top()
        still = check_retrieval_coverage(top, stage, comorbidities)
        print(f"[retrieval]   after retry, still uncovered: {still or 'none'}")

    order_by = "rerank" if FINAL_RERANK else "rrf"
    print(f"[retrieval] {len(top)} chunks fed to the generator (ordered by {order_by}):")
    for i, c in enumerate(top, 1):
        head = (c.get("heading") or "(no heading)")[:70]
        snippet = " ".join(c["text"].split())[:110]
        score = f"rerank={c['rerank_score']:.3f}" if "rerank_score" in c else f"rrf={c['rrf_score']:.4f}"
        print(f"[retrieval]   [{i}] {score} {c['source_file']} | {head}")
        print(f"[retrieval]       {snippet}...")
    return top, labeled_queries


def _format_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        src = f"{c['source_file']}" + (f" — {c['heading']}" if c.get("heading") else "")
        blocks.append(f"[{i}] (source: {src})\n{c['text']}")
    return "\n\n".join(blocks)


def generate_from_chunks(chunks, stage: str, comorbidities=None, patient_summary: str = ""):
    """Generate recommendations grounded in an explicit list of context chunks.

    Separated from retrieval so evaluation can feed a specific retrieval set
    (e.g. dense-only vs hybrid) and compare generation on identical inputs.
    """
    context = _format_context(chunks)
    como_line = f"Reported comorbidities: {', '.join(comorbidities)}." if comorbidities else "No comorbidities reported."
    user_prompt = (
        f"Predicted diabetes stage: {stage}.\n{como_line}\n\n"
        f"{patient_summary}\n\n"
        f"CONTEXT (numbered guideline excerpts)\n{context}\n\n"
        f"Write recommendations for THIS patient. Reference their specific values from the "
        f"PATIENT PROFILE above (cite the actual numbers), prioritize the abnormal findings, "
        f"and ground all clinical claims in the CONTEXT."
    )

    raw_text = _chat(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=4000)
    text, redacted = apply_drug_safety(raw_text, chunks)  # deterministic drug-grounding backstop

    sources = [
        {"n": i, "source_file": c["source_file"], "heading": c.get("heading", "")}
        for i, c in enumerate(chunks, 1)
    ]
    warnings = check_output(text, len(sources))   # structural guardrails (headings, disclaimer, doses, citations)
    if redacted:
        warnings.append(f"Redacted ungrounded drug name(s): {', '.join(redacted)}.")

    # Runtime NLI grounding (#4): flag claims the retrieved context does not entail.
    # Env-gated (default on) since it adds an NLI pass per response.
    ungrounded_claims = []
    if os.environ.get("GROUNDING_NLI", "1") == "1":
        from grounding import check_grounding
        # patient profile is also a legitimate grounding source for their own values.
        ungrounded_claims = check_grounding(text, chunks, extra_premises=[patient_summary])["ungrounded"]
        if ungrounded_claims:
            warnings.append(f"{len(ungrounded_claims)} claim(s) not entailed by the retrieved context (NLI grounding).")

    return {
        "recommendations": text,               # safe, user-facing (ungrounded drugs redacted)
        "raw_recommendations": raw_text,        # unfiltered model output (for evaluation)
        "redacted_drugs": redacted,             # any ungrounded drug names removed
        "ungrounded_claims": ungrounded_claims,  # NLI-flagged unsupported sentences
        "sources": sources,
        "context_used": len(chunks),
        "warnings": warnings,
    }


def generate_recommendations(stage: str, comorbidities=None, patient: dict = None,
                             patient_summary: str = ""):
    """Retrieve guideline context, then generate recommendations personalized to the
    patient's own values. Pass the raw form dict as `patient`; a flagged profile is
    built from it. (`patient_summary` remains for callers that supply their own text,
    e.g. the evaluation harness.)"""
    if patient is not None:
        patient_summary = summarize_patient(patient)
    chunks, queries = retrieve_context(stage, comorbidities)

    # Guardrail B: flag comorbidities that no retrieved chunk supports.
    missing = check_retrieval_coverage(chunks, stage, comorbidities)
    result = generate_from_chunks(chunks, stage, comorbidities, patient_summary)
    for c in missing:
        result["warnings"].append(f"No retrieved guideline covers '{c}' — its recommendations may lack support.")

    # Surface the formed sub-queries + retrieved chunks (for logging / audit).
    result["queries"] = [{"label": label, "query": q} for label, q in queries]
    result["retrieved"] = [{
        "source_file": c["source_file"],
        "heading": c.get("heading", ""),
        "score": c.get("rerank_score", c.get("rrf_score")),
        "text": c["text"],
    } for c in chunks]
    return result


if __name__ == "__main__":
    demo = dict(age=62, bmi=33, waist_to_hip_ratio=1.02, physical_activity_minutes_per_week=20,
                diet_score=3, alcohol_consumption_per_week=6, sleep_hours_per_day=6,
                screen_time_hours_per_day=8, systolic_bp=148, diastolic_bp=92, heart_rate=84,
                cholesterol_total=240, hdl_cholesterol=34, ldl_cholesterol=160, triglycerides=300,
                glucose_fasting=155, glucose_postprandial=230, insulin_level=22, hba1c=8.2,
                gender="Male")
    out = generate_recommendations(stage="Type 2", comorbidities=["heart failure", "chronic kidney disease"],
                                   patient=demo)
    print(out["recommendations"])
    print("\n--- SOURCES ---")
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['source_file']} | {s['heading'][:60]}")
