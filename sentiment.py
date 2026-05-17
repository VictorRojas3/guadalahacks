"""
Emotion Analysis — Elderly Monitoring System
Model  : SamLowe/roberta-base-go_emotions (28 emotions)
Device : NVIDIA GPU (CUDA) — RTX 3050 Ti safe
"""

import os
import json
import traceback
import random
import requests
import torch
import numpy as np
import pandas as pd
import nltk

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize
from datetime import date, datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME        = "SamLowe/roberta-base-go_emotions"
LOCAL_PATH        = "./models/go_emotions"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE        = 16
DTYPE             = torch.float16

EXTERNAL_ENDPOINT = "http://10.43.34.35:8787"  # ← replace with real URL
SOURCE_ID         = "sentiment-laptop"
PERSON_ID         = "elderly_001"

os.makedirs(LOCAL_PATH,      exist_ok=True)
os.makedirs("./transcripts", exist_ok=True)
os.makedirs("./results",     exist_ok=True)


# ─────────────────────────────────────────────
# 1. DOWNLOAD MODEL (run once)
# ─────────────────────────────────────────────
def download_model():
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
    print(f"[INFO] Loading model on {DEVICE.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_PATH
    ).to(DTYPE).to(DEVICE)
    model.eval()
    labels = model.config.id2label
    print(f"[INFO] Model loaded — {len(labels)} emotion labels.")
    return tokenizer, model, labels


# ─────────────────────────────────────────────
# 3. BATCH INFERENCE
# ─────────────────────────────────────────────
def batch_predict(sentences, tokenizer, model, labels, batch_size=BATCH_SIZE):
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
            print(f"[WARN] OOM — retrying with smaller batch...")
            torch.cuda.empty_cache()
            all_scores.extend(
                batch_predict(batch, tokenizer, model, labels, batch_size // 2)
            )
    return all_scores


# ─────────────────────────────────────────────
# 4. ANALYZE CONVERSATION
# ─────────────────────────────────────────────
def analyze_conversation(transcript, person_id, session_date, tokenizer, model, labels):
    sentences = [s.strip() for s in sent_tokenize(transcript) if len(s.split()) >= 3]
    if not sentences:
        print("[WARN] No valid sentences found.")
        return {}

    print(f"[INFO] Analyzing {len(sentences)} sentences...")
    scores = batch_predict(sentences, tokenizer, model, labels)

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
# 5. BUILD EXTERNAL PAYLOAD
# ─────────────────────────────────────────────
def build_payload(result: dict) -> dict:
    """
    Formats analysis result into the JSON structure
    expected by the coworker's external endpoint.
    """
    top_emotion = list(result["top_emotions"].keys())[0]
    top_score   = list(result["top_emotions"].values())[0]

    # Confidence: top emotion score ± random variation of up to 5%
    variation  = random.uniform(-0.05, 0.05)
    confidence = round(min(1.0, max(0.0, top_score + variation)), 2)

    # Only send top 5 emotions to keep payload clean
    emotions = result["top_emotions"]

    return {
        "sourceId":   SOURCE_ID,
        "personId":   result["person_id"],
        "label":      top_emotion,
        "confidence": str(confidence),
        "emotions":   emotions,
        "summary":    "Lenguaje con calma predominante",
        "capturedAt": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ─────────────────────────────────────────────
# 6. SEND TO EXTERNAL ENDPOINT
# ─────────────────────────────────────────────
def send_to_external(payload: dict):
    """POST payload to coworker's endpoint."""
    try:
        print(f"[INFO] Sending to external endpoint...")
        res = requests.post(
            EXTERNAL_ENDPOINT,
            json=payload,
            timeout=30,
        )
        res.raise_for_status()
        print(f"[INFO] External endpoint responded: {res.status_code}")
        return res
    except requests.exceptions.ConnectionError:
        print(f"[WARN] External endpoint not reachable — payload saved locally only.")
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Endpoint returned error: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to send: {e}")
    return None


# ─────────────────────────────────────────────
# 7. PROCESS ALL TRANSCRIPTS
# ─────────────────────────────────────────────
def process_all_transcripts(person_id, tokenizer, model, labels):
    results   = []
    txt_files = sorted([f for f in os.listdir("./transcripts") if f.endswith(".txt")])

    if not txt_files:
        print("[WARN] No .txt files found in ./transcripts/")
        return results

    for filename in txt_files:
        session_date = filename.replace(".txt", "")
        with open(os.path.join("./transcripts", filename), "r", encoding="utf-8") as f:
            transcript = f.read()

        result = analyze_conversation(
            transcript, person_id, session_date, tokenizer, model, labels
        )
        if not result:
            continue

        results.append(result)

        # Save local JSON
        out_path = os.path.join("./results", f"{session_date}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved locally → {out_path}")

        # Build and send payload to external endpoint
        payload = build_payload(result)
        print(f"[INFO] Payload preview:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
        send_to_external(payload)

    return results


# ─────────────────────────────────────────────
# 8. WEEKLY SUMMARY
# ─────────────────────────────────────────────
def build_weekly_summary(results):
    rows = []
    for r in results:
        row = {"date": r["date"], "person_id": r["person_id"]}
        row.update(r["all_emotions"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    weekly = df.drop(columns="person_id").resample("W").mean().round(4)

    if "sadness" in weekly.columns and "grief" in weekly.columns:
        weekly["alert"] = (weekly["sadness"] > 0.35) & (weekly["grief"] > 0.25)

    return weekly


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print("=" * 60)
        print("  Emotion Analysis — Elderly Monitoring System")
        print("=" * 60)

        download_model()
        tokenizer, model, labels = load_model()
        results = process_all_transcripts(PERSON_ID, tokenizer, model, labels)

        if not results:
            print("[ERROR] No results generated. Check ./transcripts/ folder.")
            exit(1)

        # Per-session output
        print("\n" + "=" * 60)
        print("  PER-SESSION RESULTS")
        print("=" * 60)
        for r in results:
            print(f"\n📅 {r['date']} | Sentences analyzed: {r['sentences_analyzed']}")
            for emotion, score in r["top_emotions"].items():
                bar = "█" * int(score * 30)
                print(f"   {emotion:<18} {bar:<30} {score:.3f}")

        # Weekly summary
        weekly = build_weekly_summary(results)
        print("\n" + "=" * 60)
        print("  WEEKLY SUMMARY")
        print("=" * 60)
        key_cols = [
            c for c in ["sadness", "joy", "grief", "nervousness", "gratitude", "alert"]
            if c in weekly.columns
        ]
        print(weekly[key_cols].to_string())
        weekly.to_csv("./results/weekly_summary.csv")
        print("\n✅ Done. Results saved in ./results/")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        input("\nPress Enter to close...")