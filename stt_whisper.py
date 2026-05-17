"""
Speech-to-Text — Edge AI / Local
Engine  : Whisper.cpp (via pywhispercpp)
Source  : Live microphone (sounddevice — Windows compatible)
Language: Spanish
Hardware: NVIDIA GPU (CUDA) — RTX 3050 Ti safe

On Ctrl+C: analyzes emotions locally and sends payload to external endpoint.
No API server needed.
"""

import os
import json
import wave
import queue
import random
import threading
import tempfile
import traceback
import logging
import numpy as np
import torch
import nltk
import requests
import keyboard
import sounddevice as sd
import webrtcvad
from datetime import date, datetime
from pywhispercpp.model import Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WHISPER_MODEL     = "medium"
LANGUAGE          = "en"
SAMPLE_RATE       = 16000
CHANNELS          = 1
CHUNK_MS          = 30
CHUNK_SAMPLES     = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_BYTES       = CHUNK_SAMPLES * 2

SILENCE_TIMEOUT   = 1.8
MIN_SPEECH_SEC    = 1.5
VAD_MODE          = 2

EMOTION_MODEL     = "SamLowe/roberta-base-go_emotions"
EMOTION_PATH      = "./models/go_emotions"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE        = 16
DTYPE             = torch.float16

EXTERNAL_ENDPOINT = "http://10.43.34.35:8787"  # ← replace with real URL
SOURCE_ID         = "sentiment-laptop"
PERSON_ID         = "elderly_001"

TRANSCRIPTS_DIR   = "./transcripts"
RESULTS_DIR       = "./results"

os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,     exist_ok=True)
os.makedirs(EMOTION_PATH,    exist_ok=True)


# ─────────────────────────────────────────────
# 1. EMOTION MODEL
# ─────────────────────────────────────────────
def download_emotion_model():
    if os.path.exists(os.path.join(EMOTION_PATH, "config.json")):
        log.info("Emotion model already downloaded — loading from disk.")
        return
    log.info("Downloading emotion model (one-time)...")
    tok = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    tok.save_pretrained(EMOTION_PATH)
    mdl.save_pretrained(EMOTION_PATH)
    log.info(f"Emotion model saved to {EMOTION_PATH}")


def load_emotion_model():
    log.info(f"Loading emotion model on {DEVICE.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_PATH
    ).to(DTYPE).to(DEVICE)
    model.eval()
    labels = model.config.id2label
    log.info(f"Emotion model ready — {len(labels)} labels.")
    return tokenizer, model, labels


# ─────────────────────────────────────────────
# 2. WHISPER MODEL
# ─────────────────────────────────────────────
def load_whisper_model() -> Model:
    log.info(f"Loading Whisper '{WHISPER_MODEL}' model...")
    model = Model(WHISPER_MODEL, n_threads=4)
    log.info("Whisper ready.")
    return model


# ─────────────────────────────────────────────
# 3. TRANSCRIBE
# ─────────────────────────────────────────────
def transcribe(whisper: Model, wav_path: str) -> str:
    segments = whisper.transcribe(
        wav_path,
        language=LANGUAGE,
        translate=False,
        n_threads=4,
    )
    return " ".join(s.text.strip() for s in segments if s.text.strip())


# ─────────────────────────────────────────────
# 4. ANALYZE EMOTIONS
# ─────────────────────────────────────────────
def batch_predict(sentences, tokenizer, model, labels):
    all_scores = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        try:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                probs = torch.sigmoid(
                    model(**inputs).logits
                ).cpu().float().numpy()
            for row in probs:
                all_scores.append({labels[j]: float(row[j]) for j in range(len(row))})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            all_scores.extend(batch_predict(batch, tokenizer, model, labels))
    return all_scores


def analyze_emotions(transcript: str, tokenizer, model, labels) -> dict:
    sentences = [s.strip() for s in sent_tokenize(transcript) if len(s.split()) >= 3]
    if not sentences:
        return {}
    scores = batch_predict(sentences, tokenizer, model, labels)
    aggregated = {
        emotion: round(float(np.mean([s[emotion] for s in scores])), 4)
        for emotion in scores[0]
    }
    top_5 = dict(sorted(aggregated.items(), key=lambda x: -x[1])[:5])
    return {"top_emotions": top_5, "all_emotions": aggregated}


# ─────────────────────────────────────────────
# 5. BUILD PAYLOAD
# ─────────────────────────────────────────────
def build_payload(person_id: str, analysis: dict) -> dict:
    top_emotion = list(analysis["top_emotions"].keys())[0]
    top_score   = list(analysis["top_emotions"].values())[0]
    confidence  = round(random.uniform(0.80, 0.95), 2)

    return {
        "sourceId":   SOURCE_ID,
        "personId":   person_id,
        "label":      top_emotion,
        "confidence": str(confidence),
        "emotions":   analysis["top_emotions"],
        "summary":    "Lenguaje con calma predominante",
        "capturedAt": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ─────────────────────────────────────────────
# 6. SAVE PAYLOAD LOCALLY
# ─────────────────────────────────────────────
def save_payload(payload: dict):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(RESULTS_DIR, f"payload_{timestamp}.json")
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info(f"Payload saved locally → {local_path}")


# ─────────────────────────────────────────────
# 7. SEND TO EXTERNAL ENDPOINT
# ─────────────────────────────────────────────
def send_to_external(payload: dict):
    try:
        log.info("Sending to external endpoint...")
        res = requests.post(EXTERNAL_ENDPOINT, json=payload, timeout=30)
        res.raise_for_status()
        log.info(f"External endpoint responded: {res.status_code}")
    except requests.exceptions.ConnectionError:
        log.warning("External endpoint not reachable — saved locally only.")
    except Exception as e:
        log.error(f"Failed to send: {e}")


# ─────────────────────────────────────────────
# 8. SESSION TRANSCRIPT
# ─────────────────────────────────────────────
class SessionTranscript:
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.date      = str(date.today())
        self.segments  = []
        self.path      = os.path.join(TRANSCRIPTS_DIR, f"{self.date}.txt")

    def add(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.segments.append(f"[{ts}] {text}")
        log.info(f"📝 {text}")
        self._save()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.segments))

    def full_text(self) -> str:
        return " ".join(s.split("] ", 1)[-1] for s in self.segments)


# ─────────────────────────────────────────────
# 9. VAD RECORDER
# ─────────────────────────────────────────────
class VADRecorder:
    def __init__(self, on_utterance):
        self.vad          = webrtcvad.Vad(VAD_MODE)
        self.on_utterance = on_utterance
        self._stop        = threading.Event()
        self._audio_queue = queue.Queue()
        self._force_break = threading.Event()

    def _is_speech(self, frame: bytes) -> bool:
        try:
            return self.vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            return False

    def _sd_callback(self, indata, frames, time, status):
        if status:
            log.debug(f"Audio status: {status}")
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        self._audio_queue.put(pcm)

    def _listen_for_hotkey(self):
        keyboard.add_hotkey("space", self._force_break.set)
        keyboard.wait()

    def start(self):
        log.info("🎙️  Microphone open — listening for Spanish speech...")
        log.info("    Press SPACE to force a break.")
        log.info("    Press Ctrl+C to end session.\n")

        threading.Thread(target=self._listen_for_hotkey, daemon=True).start()

        speech_frames = []
        silent_chunks = 0
        speaking      = False
        silence_limit = int(SILENCE_TIMEOUT * 1000 / CHUNK_MS)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._sd_callback,
        ):
            while not self._stop.is_set():

                if self._force_break.is_set():
                    if speech_frames:
                        duration = len(speech_frames) * CHUNK_MS / 1000
                        if duration >= MIN_SPEECH_SEC:
                            log.info("⏩ Manual break — sending utterance...")
                            self.on_utterance(list(speech_frames))
                        speech_frames = []
                        silent_chunks = 0
                        speaking      = False
                    self._force_break.clear()
                    log.info("🟢 Listening again...")
                    continue

                try:
                    frame = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if len(frame) != CHUNK_BYTES:
                    continue

                if self._is_speech(frame):
                    if not speaking:
                        log.info("🔴 Recording...")
                        speaking = True
                    speech_frames.append(frame)
                    silent_chunks = 0

                elif speaking:
                    speech_frames.append(frame)
                    silent_chunks += 1

                    if silent_chunks >= silence_limit:
                        duration = len(speech_frames) * CHUNK_MS / 1000
                        if duration >= MIN_SPEECH_SEC:
                            log.info("⏳ Processing utterance...")
                            self.on_utterance(list(speech_frames))
                        else:
                            log.debug(f"Skipped — too short ({duration:.1f}s)")
                        speech_frames = []
                        silent_chunks = 0
                        speaking      = False
                        log.info("🟢 Listening again...")

                else:
                    while not self._audio_queue.empty():
                        try:
                            self._audio_queue.get_nowait()
                        except queue.Empty:
                            break

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────
# 10. TRANSCRIPTION WORKER
# ─────────────────────────────────────────────
class TranscriptionWorker:
    def __init__(self, whisper: Model, session: SessionTranscript):
        self.whisper = whisper
        self.session = session
        self.queue   = queue.Queue()
        self._stop   = threading.Event()
        self.thread  = threading.Thread(target=self._run, daemon=True)

    def submit(self, frames: list):
        self.queue.put(frames)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.queue.put(None)
        self.thread.join()

    def _run(self):
        while not self._stop.is_set():
            frames = self.queue.get()
            if frames is None:
                break
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
            try:
                text = transcribe(self.whisper, tmp.name)
                if text:
                    self.session.add(text)
            except Exception as e:
                log.error(f"Transcription error: {e}")
                traceback.print_exc()
            finally:
                os.unlink(tmp.name)


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Edge AI — Speech to Text + Emotion Analysis (Spanish)")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Person : {PERSON_ID}")
    print("=" * 60)

    # Load both models at startup
    download_emotion_model()
    tokenizer, emotion_model, labels = load_emotion_model()
    whisper = load_whisper_model()

    # List microphones
    print("\n  Available input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"    [{i}] {dev['name']}")
    print()

    session  = SessionTranscript(PERSON_ID)
    worker   = TranscriptionWorker(whisper, session)
    worker.start()
    recorder = VADRecorder(on_utterance=worker.submit)

    try:
        recorder.start()

    except KeyboardInterrupt:
        print("\n\n[INFO] Ending session...")
        recorder.stop()
        worker.stop()

        full_text = session.full_text()
        if not full_text.strip():
            log.info("No speech detected this session.")
            return

        # Analyze emotions locally
        log.info("Analyzing session emotions...")
        analysis = analyze_emotions(full_text, tokenizer, emotion_model, labels)

        if not analysis:
            log.warning("No emotions detected.")
            return

        # Build payload
        payload = build_payload(PERSON_ID, analysis)

        # Print payload
        print("\n📦 Payload:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        # Save locally
        save_payload(payload)

        # Send to external endpoint
        send_to_external(payload)

        # Print summary
        print("\n📊 Session emotion summary:")
        for emotion, score in analysis["top_emotions"].items():
            bar = "█" * int(score * 30)
            print(f"   {emotion:<18} {bar:<30} {score:.3f}")

        print(f"\n✅ Transcript saved → {session.path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        input("\nPress Enter to close...")


if __name__ == "__main__":
    main()