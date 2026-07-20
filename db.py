"""Simple SQLite audit log: one row per /recommend call.

Persists the patient profile (the 17 clinical fields), the formed retrieval
sub-queries, and the retrieved chunks — for audit, debugging, and review.
No external dependency (sqlite3 is in the stdlib); the DB is a single file.

Inspect it:
    sqlite3 recommendation_log.db "SELECT id, timestamp, stage FROM recommendation_log;"
    sqlite3 recommendation_log.db "SELECT queries FROM recommendation_log WHERE id=1;"
"""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Default: project root. RECOMMEND_DB overrides it (Docker mounts a volume there).
DB_PATH = Path(os.environ.get("RECOMMEND_DB",
                              Path(__file__).resolve().parent / "recommendation_log.db"))

# Patient profile fields persisted per request (as requested).
PROFILE_FIELDS = [
    "age", "bmi", "hba1c", "glucose_fasting", "glucose_postprandial",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
    "systolic_bp", "diastolic_bp", "heart_rate", "physical_activity_minutes_per_week",
    "sleep_hours_per_day", "diet_score", "alcohol_consumption_per_week", "insulin_level",
]


def init_db():
    """Create the log table if it doesn't exist (idempotent)."""
    profile_cols = ",\n                ".join(f"{f} REAL" for f in PROFILE_FIELDS)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS recommendation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stage TEXT,
                comorbidities TEXT,          -- JSON array
                {profile_cols},
                queries TEXT,                -- JSON: [{{label, query}}]
                retrieval TEXT               -- JSON: [{{source_file, heading, score, text}}]
            )
        """)


def log_recommendation(patient, stage, comorbidities, queries, retrieval):
    """Insert one audit row. Returns the new row id."""
    cols = ["timestamp", "stage", "comorbidities"] + PROFILE_FIELDS + ["queries", "retrieval"]
    values = (
        [datetime.now().isoformat(timespec="seconds"), stage, json.dumps(comorbidities or [])]
        + [patient.get(f) for f in PROFILE_FIELDS]
        + [json.dumps(queries, ensure_ascii=False), json.dumps(retrieval, ensure_ascii=False)]
    )
    placeholders = ", ".join(["?"] * len(cols))
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"INSERT INTO recommendation_log ({', '.join(cols)}) VALUES ({placeholders})", values)
        return cur.lastrowid


init_db()  # ensure the table exists on import
