import os
import json
import torch
import numpy as np
import nltk
import pandas as pd
from datetime import date
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
# ─────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize
 
MODEL_NAME = "pysentimiento/robertuito-emotion-analysis"
LOCAL_PATH = "./models/go_emotions"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16       # safe for 4GB VRAM with float16
DTYPE      = torch.float16
 
os.makedirs(LOCAL_PATH,       exist_ok=True)
os.makedirs("./transcripts",  exist_ok=True)
os.makedirs("./results",      exist_ok=True)
 
 
# ─────────────────────────────────────────────
# 1. DOWNLOAD & SAVE MODEL (run once)
# ─────────────────────────────────────────────
def download_model():
    """Download model from HuggingFace and persist locally for offline use."""
    if os.path.exists(os.path.join(LOCAL_PATH, "config.json")):
        print("[INFO] Model already downloaded — loading from disk.")
        return
 
    print("[INFO] Downloading model (one-time)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tok.save_pretrained(LOCAL_PATH)
    mdl.save_pretrained(LOCAL_PATH)
    print(f"[INFO] Model saved to {LOCAL_PATH}")
 
 
# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    """Load tokenizer and model onto GPU with float16."""
    print(f"[INFO] Loading model on {DEVICE.upper()} ({'GPU ✓' if DEVICE == 'cuda' else 'CPU — slower'})...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_PATH,
        dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    labels = model.config.id2label  # {0: 'admiration', 1: 'amusement', ...}
    print(f"[INFO] Model loaded — {len(labels)} emotion labels available.")
    return tokenizer, model, labels
 
 
# ─────────────────────────────────────────────
# 3. BATCH INFERENCE (with OOM guard)
# ─────────────────────────────────────────────
def batch_predict(sentences: list[str], tokenizer, model, labels, batch_size=BATCH_SIZE) -> list[dict]:
    """
    Run sigmoid multi-label classification in batches.
    Automatically halves batch size on CUDA OOM.
    """
    all_scores = []
 
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        try:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)
 
            with torch.no_grad():
                logits = model(**inputs).logits
                probs  = torch.sigmoid(logits).cpu().float().numpy()
 
            for row in probs:
                all_scores.append({labels[j]: float(row[j]) for j in range(len(row))})
 
        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] OOM at batch_size={batch_size} — retrying with {batch_size // 2}...")
            torch.cuda.empty_cache()
            all_scores.extend(
                batch_predict(batch, tokenizer, model, labels, batch_size=batch_size // 2)
            )
 
    return all_scores
 
 
# ─────────────────────────────────────────────
# 4. ANALYZE A SINGLE CONVERSATION
# ─────────────────────────────────────────────
def analyze_conversation(
    transcript: str,
    person_id: str,
    session_date,
    tokenizer, model, labels
) -> dict:
    """
    Tokenize transcript into sentences, run batch inference,
    and return mean emotion scores for the full session.
    """
    sentences = [s.strip() for s in sent_tokenize(transcript) if len(s.split()) >= 3]
 
    if not sentences:
        print("[WARN] No valid sentences found in transcript.")
        return {}
 
    print(f"[INFO] Analyzing {len(sentences)} sentences for {person_id} on {session_date}...")
    scores = batch_predict(sentences, tokenizer, model, labels)
 
    # Mean score per emotion across all sentences
    aggregated = {
        emotion: round(float(np.mean([s[emotion] for s in scores])), 4)
        for emotion in scores[0]
    }
 
    top_5 = dict(sorted(aggregated.items(), key=lambda x: -x[1])[:5])
 
    return {
        "person_id":          person_id,
        "date":               str(session_date),
        "top_emotions":       top_5,
        "all_emotions":       aggregated,
        "sentences_analyzed": len(sentences),
    }
 
 
# ─────────────────────────────────────────────
# 5. PROCESS ALL TRANSCRIPTS IN A FOLDER
# ─────────────────────────────────────────────
def process_all_transcripts(person_id: str, tokenizer, model, labels) -> list[dict]:
    """
    Read every .txt file in ./transcripts/, run analysis,
    and save per-session JSON to ./results/.
    """
    results = []
 
    txt_files = sorted([f for f in os.listdir("./transcripts") if f.endswith(".txt")])
    if not txt_files:
        print("[WARN] No .txt files found in ./transcripts/")
        return results
 
    for filename in txt_files:
        session_date = filename.replace(".txt", "")
        filepath     = os.path.join("./transcripts", filename)
 
        with open(filepath, "r", encoding="utf-8") as f:
            transcript = f.read()
 
        result = analyze_conversation(transcript, person_id, session_date, tokenizer, model, labels)
        if not result:
            continue
 
        results.append(result)
 
        out_path = os.path.join("./results", f"{session_date}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved → {out_path}")
 
    return results
 
 
# ─────────────────────────────────────────────
# 6. WEEKLY DASHBOARD AGGREGATION
# ─────────────────────────────────────────────
def build_weekly_summary(results: list[dict]) -> pd.DataFrame:
    """
    Resample daily emotion scores into weekly means.
    Adds an alert flag for sustained sadness + grief.
    """
    rows = []
    for r in results:
        row = {"date": r["date"], "person_id": r["person_id"]}
        row.update(r["all_emotions"])
        rows.append(row)
 
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
 
    weekly = df.drop(columns="person_id").resample("W").mean().round(4)
 
    # Alert: sadness AND grief above threshold for the week
    if "sadness" in weekly.columns and "grief" in weekly.columns:
        weekly["alert"] = (weekly["sadness"] > 0.35) & (weekly["grief"] > 0.25)
 
    return weekly
 
 
# ─────────────────────────────────────────────
# 7. EXAMPLE TEST
# ─────────────────────────────────────────────
EXAMPLE_TRANSCRIPTS = {
    "2024-01-08.txt": """
        I woke up quite early again, couldn't really get back to sleep.
        Mary called me this morning, that was a lovely surprise.
        I miss seeing her in person though, it's been months since she visited.
        The garden looks beautiful today, I spent some time outside.
        I'm not sure what I'll have for dinner, maybe I'll just have soup again.
        My knee has been bothering me, makes it hard to walk to the mailbox.
        At least the weather was nice today, I sat by the window for a while.
    """,
    "2024-01-15.txt": """
        Another night of poor sleep, I keep thinking about Thomas.
        I tried to call my son but he didn't answer, I'll try again later.
        I'm grateful for the meals that get delivered, saves me the trouble.
        The house feels very quiet today, I turned on the radio for company.
        I watched some birds outside, a red one sat on the fence for a while.
        I'm a little worried about my appointment next week, hope it goes fine.
        Feeling a bit tired today but nothing serious, just the usual.
    """,
    "2024-01-22.txt": """
        I slept much better last night, what a difference that makes.
        My granddaughter came to visit today! She brought flowers, yellow ones.
        We had tea together and looked at old photographs, it was wonderful.
        I laughed more today than I have in weeks, she's such a joy.
        We went for a short walk around the block, my knee held up nicely.
        I feel hopeful today, like things are going to be alright.
        I made a real dinner tonight, pasta with tomatoes from the garden.
    """,
}
 
def create_example_transcripts():
    """Write example .txt files to ./transcripts/ for testing."""
    for filename, content in EXAMPLE_TRANSCRIPTS.items():
        path = os.path.join("./transcripts", filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip())
    print(f"[INFO] Created {len(EXAMPLE_TRANSCRIPTS)} example transcripts in ./transcripts/")
 
 
# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Emotion Analysis — Elderly Monitoring System")
    print("=" * 60)
 
    # Step 1 — ensure model is local
    download_model()
 
    # Step 2 — load model
    tokenizer, model, labels = load_model()
 
    # Step 3 — create example transcripts for testing
    create_example_transcripts()
 
    # Step 4 — process all transcripts
    results = process_all_transcripts("elderly_001", tokenizer, model, labels)
 
    if not results:
        print("[ERROR] No results generated. Check ./transcripts/ folder.")
        exit(1)
 
    # Step 5 — print per-session top emotions
    print("\n" + "=" * 60)
    print("  PER-SESSION RESULTS")
    print("=" * 60)
    for r in results:
        print(f"\n📅 {r['date']} | Sentences analyzed: {r['sentences_analyzed']}")
        for emotion, score in r["top_emotions"].items():
            bar = "█" * int(score * 30)
            print(f"   {emotion:<18} {bar:<30} {score:.3f}")
 
    # Step 6 — weekly summary
    weekly = build_weekly_summary(results)
    print("\n" + "=" * 60)
    print("  WEEKLY SUMMARY (key emotions)")
    print("=" * 60)
    key_cols = [c for c in ["sadness", "joy", "grief", "nervousness", "gratitude", "alert"] if c in weekly.columns]
    print(weekly[key_cols].to_string())
 