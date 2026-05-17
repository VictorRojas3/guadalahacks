"""
Emotion Analysis — Local REST API Server
Stack: FastAPI + Uvicorn
Serves emotion data to a React/Vue/JS dashboard on the same machine or local network.
"""

import os
import json
import torch
import numpy as np
import nltk
from datetime import date, datetime
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOCAL_PATH  = "./models/go_emotions"
RESULTS_DIR = "./results"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 16
DTYPE       = torch.float16

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# GLOBAL MODEL STATE
# ─────────────────────────────────────────────
tokenizer = None
model     = None
labels    = None


# ─────────────────────────────────────────────
# LIFESPAN — load model once at startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, labels
    print(f"[INFO] Loading model on {DEVICE.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_PATH, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    labels = model.config.id2label
    print(f"[INFO] Model ready — {len(labels)} emotion labels.")
    yield
    print("[INFO] Shutting down.")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Emotion Analysis API",
    description="Local REST API for elderly emotion tracking dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any local origin (React dev server, file://, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict to your dashboard URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    person_id:  str
    transcript: str
    session_date: Optional[str] = None   # "YYYY-MM-DD", defaults to today


class AnalyzeResponse(BaseModel):
    person_id:          str
    date:               str
    top_emotions:       dict
    all_emotions:       dict
    sentences_analyzed: int


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
def batch_predict(sentences: list[str], batch_size: int = BATCH_SIZE) -> list[dict]:
    all_scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        try:
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                probs = torch.sigmoid(model(**inputs).logits).cpu().float().numpy()
            for row in probs:
                all_scores.append({labels[j]: float(row[j]) for j in range(len(row))})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            all_scores.extend(batch_predict(batch, batch_size=batch_size // 2))
    return all_scores


def run_analysis(transcript: str, person_id: str, session_date: str) -> dict:
    sentences = [s.strip() for s in sent_tokenize(transcript) if len(s.split()) >= 3]
    if not sentences:
        raise ValueError("No valid sentences found in transcript.")

    scores     = batch_predict(sentences)
    aggregated = {
        emotion: round(float(np.mean([s[emotion] for s in scores])), 4)
        for emotion in scores[0]
    }
    top_5 = dict(sorted(aggregated.items(), key=lambda x: -x[1])[:5])

    result = {
        "person_id":          person_id,
        "date":               session_date,
        "top_emotions":       top_5,
        "all_emotions":       aggregated,
        "sentences_analyzed": len(sentences),
    }

    # Persist to disk
    out_path = os.path.join(RESULTS_DIR, f"{person_id}_{session_date}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Check that the server and model are up."""
    return {
        "status": "ok",
        "device": DEVICE,
        "model":  LOCAL_PATH,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Submit a transcript for emotion analysis.
    Returns aggregated emotion scores for the session.
    """
    session_date = req.session_date or str(date.today())
    try:
        result = run_analysis(req.transcript, req.person_id, session_date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{person_id}")
def get_all_results(person_id: str):
    """
    Return all saved session results for a given person,
    sorted by date ascending — ready for chart rendering.
    """
    files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(person_id) and f.endswith(".json")
    ]
    if not files:
        raise HTTPException(status_code=404, detail=f"No results found for '{person_id}'.")

    sessions = []
    for filename in sorted(files):
        with open(os.path.join(RESULTS_DIR, filename)) as f:
            sessions.append(json.load(f))

    return {"person_id": person_id, "sessions": sessions}


@app.get("/results/{person_id}/{session_date}")
def get_session(person_id: str, session_date: str):
    """Return emotion scores for a single session date."""
    path = os.path.join(RESULTS_DIR, f"{person_id}_{session_date}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Session not found.")
    with open(path) as f:
        return json.load(f)


@app.get("/results/{person_id}/weekly/summary")
def weekly_summary(person_id: str):
    """
    Return weekly averaged emotion scores across all sessions.
    Includes an alert flag for high sadness + grief weeks.
    """
    import pandas as pd

    files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(person_id) and f.endswith(".json")
    ]
    if not files:
        raise HTTPException(status_code=404, detail=f"No results found for '{person_id}'.")

    rows = []
    for filename in sorted(files):
        with open(os.path.join(RESULTS_DIR, filename)) as f:
            data = json.load(f)
        row = {"date": data["date"]}
        row.update(data["all_emotions"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    weekly = df.resample("W").mean().round(4)

    if "sadness" in weekly.columns and "grief" in weekly.columns:
        weekly["alert"] = (weekly["sadness"] > 0.35) & (weekly["grief"] > 0.25)

    weekly.index = weekly.index.strftime("%Y-%m-%d")
    return {"person_id": person_id, "weekly": weekly.to_dict(orient="index")}


@app.get("/persons")
def list_persons():
    """List all person IDs that have saved results."""
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    ids   = sorted(set(f.rsplit("_", 1)[0] for f in files))
    return {"persons": ids}


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",   # accessible on local network
        port=8000,
        reload=False,      # set True during development
    )
