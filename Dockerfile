# =============================================================================
# Dockerfile — Resilient AI benchmark + dataset build
# Base : PyTorch 2.5.1 + CUDA 12.1 runtime (compatible RunPod GPU)
#
# Image auto-suffisante : toutes les libs sont installées dans l'image
# (torch pinné, llama-cpp-python CUDA, gptqmodel, compressed-tensors, autoawq,
# LD_LIBRARY_PATH résolu via ldconfig, HF_HOME pré-configuré sur /workspace).
# → Au runtime, AUCUN pip install ni export manuel requis.
#
# Build via GitHub Actions à chaque push (cf. .github/workflows/docker-publish.yml).
# Image disponible sur : ghcr.io/<user>/<repo>:latest
#
# Run sur Runpod : utiliser l'image en template, monter /workspace (persistant),
# l'entrypoint crée /workspace/hf-cache + symlinks /app/{results,reports}.
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ---- Variables d'env globales -----------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CODECARBON_LOG_LEVEL=error

# Cache HF + dossiers de travail sur /workspace (Network Volume Runpod).
# L'entrypoint.sh créera ces dossiers au runtime (mkdir -p).
ENV HF_HOME=/workspace/hf-cache \
    HF_HUB_CACHE=/workspace/hf-cache/hub \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    TMPDIR=/workspace/tmp \
    PIP_CACHE_DIR=/workspace/pip-cache

# ---- Apt deps ---------------------------------------------------------------
# ca-certificates : SSL pour HF Hub
# git : pull du code depuis GitHub si besoin (procédure README "Reprise rapide")
# build-essential + gcc + g++ : compilation de pypcre (dep de gptqmodel)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        gcc \
        g++ \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Couche heavy : deps Python (~5 GB) -------------------------------------
# Mise en cache séparée du code : tant que requirements.txt ne change pas,
# cette couche est réutilisée → rebuild incrémental rapide.
COPY requirements.txt .

# 1) Wheel CUDA 12.1 pré-compilé pour llama-cpp-python (backend GGUF).
#    Gain ~10 min vs source build.
RUN pip install --upgrade pip && \
    pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 2) Force torch + torchvision aux versions compatibles AVANT le reste.
#    L'image de base a déjà torch 2.5.1 + cuda 12.1, on ré-installe explicitement
#    pour s'assurer que rien n'a été modifié et que torchvision est aligné.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 3) Toutes les autres deps depuis requirements.txt (transformers, datasets,
#    codecarbon, bitsandbytes, compressed-tensors, etc.). NB : autoawq et
#    gptqmodel sont installes en best-effort apres (etapes risquees).
RUN pip install -r requirements.txt

# 3bis) autoawq et gptqmodel : best-effort (ces packages compilent du C/C++
#       et peuvent echouer sur GHA ou des envs serres). En cas d'echec, le
#       build continue sans eux ; seuls les modeles AWQ Chunity / certains
#       formats AWQ legacy seront indisponibles a l'execution.
RUN pip install autoawq || echo "WARNING: autoawq install failed - AWQ legacy models will not load"
RUN pip install gptqmodel || echo "WARNING: gptqmodel install failed - Chunity AWQ models will not load"

# 4) Sanity check au build : torch+torchvision matchent et l'opérateur nms
#    existe (sinon le build échoue au lieu de planter au runtime).
#    Print AVANT assert pour que les logs GHA montrent la version réelle
#    en cas d'échec (debug plus rapide).
RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__); from torchvision.io import decode_image; print('decode_image import OK'); assert torch.__version__.startswith('2.5.1'), torch.__version__; assert torchvision.__version__.startswith('0.20'), torchvision.__version__"

# ---- Résolution des libs CUDA pour llama-cpp-python -------------------------
# llama-cpp-python compile en CUDA cherche libcudart/libcublas dans
# LD_LIBRARY_PATH. Les libs sont fournies par les wheels nvidia/* installés
# par torch dans /opt/conda/lib/python3.11/site-packages/nvidia/*/lib mais
# ne sont pas dans LD_LIBRARY_PATH par défaut → on les ajoute via ldconfig
# (méthode propre Linux qui survit aux nouvelles sessions).
RUN find /opt/conda/lib/python3.11/site-packages/nvidia -name lib -type d > /etc/ld.so.conf.d/nvidia-conda.conf && ldconfig && python -c "from llama_cpp import Llama; print('llama_cpp loadable')"

# ---- Couche light : code projet (relayered à chaque commit) -----------------
COPY . .

RUN chmod +x scripts/*.sh

# ---- Entrypoint : crée /workspace/* + symlinks au start du container --------
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Default : sleep infinity → permet SSH dans le pod sans que le container exit.
# Override possible via Runpod template ou docker run :
#   docker run ... resilient-ai bash scripts/bench_safe.sh
CMD ["sleep", "infinity"]
