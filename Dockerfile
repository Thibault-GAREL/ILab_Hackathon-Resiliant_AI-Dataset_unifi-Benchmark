# =============================================================================
# Dockerfile — Resilient AI benchmark + dataset build
# Base : PyTorch 2.5.1 + CUDA 12.1 runtime (compatible RunPod GPU)
#
# Build :
#   docker build -t resilient-ai:latest .
#
# Run local (avec GPU + cache HF persistant) :
#   docker run --gpus all \
#       -v $PWD/results:/app/results \
#       -v $PWD/reports:/app/reports \
#       -v hf-cache:/cache/huggingface \
#       -e HF_TOKEN=hf_xxx \
#       resilient-ai:latest \
#       python scripts/bench_all.py --label gemma3n-e4b-bf16
#
# Sur Runpod : push l'image sur Docker Hub puis utilise-la dans un template
# (cf. README.md section "Workflow Docker / Runpod").
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Cache HF redirigé vers un volume montable (évite de saturer le container disk
# et permet de réutiliser les téléchargements de modèles entre runs)
ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    CODECARBON_LOG_LEVEL=error

# ca-certificates : SSL pour HF Hub.
# git : nécessaire pour re-pull le code à chaque restart du pod (le code dans
#       /app est dans le container ephemere, contrairement à /workspace qui
#       persiste — cf. README section "Reprise rapide").
# build-essential + gcc : nécessaires pour pip install gptqmodel (compile
#       pypcre depuis les sources). Sans ça, "FileNotFoundError: 'gcc'".
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        gcc \
        g++ \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Couche heavy : deps Python ---------------------------------------------
# Mise en cache séparée du code : tant que requirements.txt ne change pas,
# cette couche est réutilisée → rebuild incrémental rapide.
COPY requirements.txt .

# Wheel CUDA 12.1 pré-compilé pour llama-cpp-python (gain ~10 min vs source build).
# Réservé pour V2 backend GGUF — pas utilisé par le pipeline actuel mais déjà
# disponible dans l'image pour ne pas avoir à rebuild plus tard.
RUN pip install --upgrade pip && \
    pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 && \
    pip install -r requirements.txt && \
    pip install autoawq

# ---- Couche light : code projet ---------------------------------------------
# Ce COPY se réinvalide dès qu'un fichier source change → mais comme c'est en
# dernier, seules les ~MB de code sont relayered.
COPY . .

# Volumes recommandés au runtime :
#   /cache/huggingface  : cache modèles HF (persiste entre runs, évite re-DL)
#   /app/results        : registry.csv + artefacts par run (tu veux persister)
#   /app/reports        : rapports JSON eval_loop (tu veux persister)
RUN mkdir -p /cache/huggingface /app/results/runs /app/reports

# Default : liste les modèles disponibles. Override avec :
#   docker run ... resilient-ai python scripts/bench_all.py --all
CMD ["python", "scripts/bench_all.py", "--list"]
