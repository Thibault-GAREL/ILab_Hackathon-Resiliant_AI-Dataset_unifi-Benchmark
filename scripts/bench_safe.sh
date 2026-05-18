#!/bin/bash
# =============================================================================
# Wrapper Runpod pour scripts/bench_all.py
#
# Responsabilités exclusives :
#   1. Forcer HF_HOME vers /workspace/hf-cache (Network Volume persistant)
#      sinon le cache atterrit sur /cache/huggingface du container ephemere
#      (5 GB max) qui sature au 1er modele.
#   2. Symlink /app/results et /app/reports vers /workspace pour que le
#      registry survive un stop/start du pod.
#   3. Tout le reste (lecture YAML, dispatch backend transformers/llama_cpp,
#      cleanup cache, skip_existing) est délégué à bench_all.py.
#
# Variables d'env optionnelles :
#   DATASET_LOCAL  (défaut: /workspace/code/data/unified)
#   COUNTRY        (défaut: CZE - Runpod EU-CZ-1)
#   LIMIT          (défaut: 20 samples/source = ~5-10 min/modèle)
#                  Mettre vide ('') pour passer le full subsample du YAML.
#
# Usage :
#   bash scripts/bench_safe.sh                       # tous les modèles
#   LIMIT=100 bash scripts/bench_safe.sh             # plus de samples
#   nohup bash scripts/bench_safe.sh > /workspace/bench.log 2>&1 & disown
# =============================================================================
set -u

# ----- 1. Force HF cache vers /workspace ------------------------------------
if [ -d /workspace ]; then
    export HF_HOME=/workspace/hf-cache
    export HF_HUB_CACHE=/workspace/hf-cache/hub
    export TRANSFORMERS_CACHE=/workspace/hf-cache
    mkdir -p /workspace/hf-cache/hub
    echo "[bench_safe] HF_HOME=$HF_HOME"
else
    echo "⚠ /workspace introuvable — HF cache restera sur le container disk."
fi

# ----- 1bis. CUDA libs pour llama-cpp-python (backend GGUF) -----------------
# llama-cpp-python compile avec CUDA cherche libcudart.so.12 / libcublas.so.12
# dans LD_LIBRARY_PATH. Sur l'image PyTorch runtime, ces libs sont fournies
# par les wheels nvidia/* dans /opt/conda/lib/python3.11/site-packages/nvidia/*/lib
# mais ne sont pas dans LD_LIBRARY_PATH par défaut → ajout systématique ici.
NVIDIA_LIB_ROOT=/opt/conda/lib/python3.11/site-packages/nvidia
if [ -d "$NVIDIA_LIB_ROOT" ]; then
    NVIDIA_LIBS=$(find "$NVIDIA_LIB_ROOT" -name "*.so*" 2>/dev/null \
        | xargs -I {} dirname {} | sort -u | tr '\n' ':')
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
    echo "[bench_safe] LD_LIBRARY_PATH set ($(echo "$NVIDIA_LIBS" | tr ':' '\n' | wc -l) dossiers nvidia)"
fi

# ----- 1ter. TMPDIR vers /workspace pour pip et caches divers ---------------
# /tmp est sur le container disk (5 GB max) → un download torch sature.
if [ -d /workspace ]; then
    export TMPDIR=/workspace/tmp
    export PIP_CACHE_DIR=/workspace/pip-cache
    mkdir -p /workspace/tmp /workspace/pip-cache
fi

# ----- 2. Symlink results/ et reports/ vers /workspace ----------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for d in results reports; do
    target="/workspace/${d}"
    local_path="${PROJECT_ROOT}/${d}"
    if [ -d /workspace ] && [ ! -L "$local_path" ]; then
        mkdir -p "$target/runs"
        [ -d "$local_path" ] && rm -rf "$local_path"
        ln -s "$target" "$local_path"
        echo "[bench_safe] symlink: $local_path -> $target"
    fi
done

# ----- 3. Délégation à bench_all.py -----------------------------------------
DATASET_LOCAL=${DATASET_LOCAL:-/workspace/code/data/unified}
COUNTRY=${COUNTRY:-CZE}
LIMIT=${LIMIT:-20}

CMD=(
    python "${PROJECT_ROOT}/scripts/bench_all.py"
    --all
    --skip_existing
    --dataset_local "$DATASET_LOCAL"
    --country_iso "$COUNTRY"
    --cleanup_cache_after
)

# Si LIMIT vide, on n'ajoute pas le flag (laisse le YAML défaut = full)
if [ -n "$LIMIT" ]; then
    CMD+=(--limit_per_source "$LIMIT")
fi

echo "[bench_safe] cmd: ${CMD[*]}"
cd "$PROJECT_ROOT"
"${CMD[@]}"
