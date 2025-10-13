# =========================================================
# 🧠 Base environment — lightweight Python 3.10 image
# =========================================================
FROM python:3.10-slim AS base
LABEL maintainer="Siddharth Chalasani <162922611+sidc26@users.noreply.github.com>"
WORKDIR /app

# --- System dependencies (OpenCV, Pillow, ffmpeg) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --- Core build tools ---
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Common Python deps for both CPU and GPU builds ---
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas \
    pywaggle[vision]==0.56.0

# =========================================================
# 🧩 CPU build stage (for local testing on laptop)
# =========================================================
FROM base AS cpu
RUN echo "Installing CPU PyTorch stack..." && \
    pip install --no-cache-dir \
        torch==2.2.0 \
        torchvision==0.17.0 \
        torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cpu

# =========================================================
# ⚙️ GPU build stage (for Jetson Orin)
# =========================================================
FROM base AS arm64
RUN echo "Installing Jetson-compatible PyTorch stack..." && \
    pip install --no-cache-dir \
      --extra-index-url https://pypi.ngc.nvidia.com \
      torch==2.3.0 \
      torchvision==0.18.0 \
      torchaudio==2.3.0

# =========================================================
# 🚀 Final runtime image (selects stage at build time)
# =========================================================
FROM arm64 AS final
COPY . .
RUN mkdir -p weights test-run
ENTRYPOINT ["python3", "main.py"]
