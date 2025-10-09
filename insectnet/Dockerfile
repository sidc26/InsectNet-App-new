# =========================================================
# 🧠 Base environment — lightweight Python 3.10 image
# =========================================================
FROM python:3.10-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pywaggle[vision]==0.56.0 \
    pandas

# =========================================================
# 🧩 CPU build for local testing
# =========================================================
FROM base AS cpu
RUN echo "Installing CPU PyTorch stack..." && \
    pip install --no-cache-dir \
        torch==2.2.0 \
        torchvision==0.17.0 \
        torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cpu

# =========================================================
# 🚀 Final runtime image
# =========================================================
FROM cpu AS final
COPY . .
ENTRYPOINT ["python3", "main.py"]
