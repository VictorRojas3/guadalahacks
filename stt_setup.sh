#!/bin/bash
# ─────────────────────────────────────────────
# Whisper.cpp Setup Script — Edge AI / Local STT
# Spanish, Live Microphone, NVIDIA GPU (CUDA)
# ─────────────────────────────────────────────

set -e

echo "=============================="
echo "  Whisper.cpp Setup"
echo "=============================="

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y \
    git build-essential cmake \
    libsdl2-dev libasound2-dev \
    ffmpeg curl

# 2. Clone whisper.cpp
echo "[2/5] Cloning whisper.cpp..."
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp

# 3. Build with CUDA support (NVIDIA GPU)
echo "[3/5] Building with CUDA support..."
cmake -B build \
    -DWHISPER_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 4. Download Spanish-optimized model
echo "[4/5] Downloading Whisper medium model (best for Spanish)..."
mkdir -p models
bash ./models/download-ggml-model.sh medium
# Alternatives by VRAM:
#   tiny   (~75MB)  — fastest, lower accuracy
#   base   (~142MB) — good balance for simple speech
#   small  (~466MB) — recommended minimum for Spanish elderly speech
#   medium (~1.5GB) — best accuracy on 4GB VRAM  ← DEFAULT
#   large  (~3GB)   — highest accuracy, tight on 4GB VRAM

echo "[5/5] Installing Python dependencies..."
pip install \
    pyaudio \
    numpy \
    requests \
    wave \
    webrtcvad \
    pywhispercpp

echo ""
echo "✅ Setup complete!"
echo "   Run: python stt_whisper.py"
