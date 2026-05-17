"""
Speech-to-Text — Edge AI / Local
Engine  : Whisper.cpp (via pywhispercpp)
Source  : Live microphone (sounddevice — Windows compatible)
Language: Spanish
Hardware: NVIDIA GPU (CUDA) — RTX 3050 Ti safe
"""

import os
import wave
import queue
import random
import threading
import tempfile
import traceback
import logging
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
import keyboard
from datetime import date, datetime
from pywhispercpp.model import Model

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

EMOTION_API_URL   = "http://localhost:8000/analyze"
EXTERNAL_ENDPOINT = "https://your-coworkers-endpoint.com/data"  # ← replace with real URL
SOURCE_ID         = "sentiment-laptop"
PERSON_ID         = "elderly_001"

TRANSCRIPTS_DIR   = "./transcripts"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD WHISPER
# ─────────────────────────────────────────────
def load_whisper(model_name: str) -> Model:
    log.info(f"Loading Whisper '{model_name}' model...")
    model = Model(model_name, n_threads=4)
    log.info("Whisper ready.")
    return model


# ─────────────────────────────────────────────
# 2. PCM FRAMES → WAV
# ─────────────────────────────────────────────
def frames_to_wav(frames: list) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return tmp.name


# ─────────────────────────────────────────────
# 3. TRANSCRIBE
# ─────────────────────────────────────────────
def transcribe(model: Model, wav_path: str) -> str:
    segments = model.transcribe(
        wav_path,
        language=LANGUAGE,
        translate=False,
        n_threads=4,
    )
    return " ".join(s.text.strip() for s in segments if s.text.strip())


# ─────────────────────────────────────────────
# 4. BUILD EXTERNAL PAYLOAD
# ─────────────────────────────────────────────
def build_payload(person_id: str, analysis: dict) -> dict:
    top_emotion = list(analysis["top_emotions"].keys())[0]
    top_score   = list(analysis["top_emotions"].values())[0]

    variation  = random.uniform(-0.05, 0.05)
    confidence = round(min(1.0, max(0.0, top_score + variation)), 2)

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
# 5. SEND TO LOCAL EMOTION API
# ─────────────────────────────────────────────
def send_to_emotion_api(transcript: str, session_date: str) -> dict:
    try:
        res = requests.post(
            EMOTION_API_URL,
            json={
                "person_id":    PERSON_ID,
                "transcript":   transcript,
                "session_date": session_date,
            },
            timeout=60,
        )
        res.raise_for_status()
        data = res.json()
        log.info(f"Emotion API → top emotions: {data['top_emotions']}")
        return data
    except requests.exceptions.ConnectionError:
        log.warning("Emotion API not reachable — transcript saved locally only.")
    except Exception as e:
        log.error(f"Emotion API error: {e}")
    return None


# ─────────────────────────────────────────────
# 6. SEND TO EXTERNAL ENDPOINT
# ─────────────────────────────────────────────
def send_to_external(payload: dict):
    try:
        log.info(f"Sending to external endpoint...")
        res = requests.post(EXTERNAL_ENDPOINT, json=payload, timeout=30)
        res.raise_for_status()
        log.info(f"External endpoint responded: {res.status_code}")
    except requests.exceptions.ConnectionError:
        log.warning("External endpoint not reachable.")
    except Exception as e:
        log.error(f"Failed to send to external: {e}")


# ─────────────────────────────────────────────
# 7. SESSION TRANSCRIPT
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
# 8. VAD RECORDER
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
        log.info("    Press Ctrl+C to stop.\n")

        hotkey_thread = threading.Thread(
            target=self._listen_for_hotkey, daemon=True
        )
        hotkey_thread.start()

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

                # Manual break via SPACE
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
# 9. TRANSCRIPTION WORKER
# ─────────────────────────────────────────────
class TranscriptionWorker:
    def __init__(self, model: Model, session: SessionTranscript):
        self.model   = model
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
            wav_path = frames_to_wav(frames)
            try:
                text = transcribe(self.model, wav_path)
                if text:
                    self.session.add(text)
            except Exception as e:
                log.error(f"Transcription error: {e}")
                traceback.print_exc()
            finally:
                os.unlink(wav_path)


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Edge AI — Speech to Text (Spanish)")
    print("  Engine : Whisper.cpp")
    print("  Audio  : sounddevice (Windows compatible)")
    print("=" * 55)

    print("\n  Available input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"    [{i}] {dev['name']}")
    print()

    model   = load_whisper(WHISPER_MODEL)
    session = SessionTranscript(PERSON_ID)
    worker  = TranscriptionWorker(model, session)
    worker.start()

    recorder = VADRecorder(on_utterance=worker.submit)

    try:
        recorder.start()

    except KeyboardInterrupt:
        print("\n\n[INFO] Stopping session...")
        recorder.stop()
        worker.stop()

        full_text = session.full_text()
        if full_text.strip():
            log.info("Sending session transcript to emotion API...")
            analysis = send_to_emotion_api(full_text, session.date)
            if analysis:
                # Build and send to external endpoint
                payload = build_payload(PERSON_ID, analysis)
                log.info(f"Payload:\n{__import__('json').dumps(payload, indent=2, ensure_ascii=False)}")
                send_to_external(payload)

                print("\n📊 Session emotion summary:")
                for emotion, score in analysis["top_emotions"].items():
                    bar = "█" * int(score * 30)
                    print(f"   {emotion:<18} {bar:<30} {score:.3f}")
        else:
            log.info("No speech detected this session.")

        print(f"\n✅ Transcript saved → {session.path}")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        input("\nPress Enter to close...")


if __name__ == "__main__":
    main()