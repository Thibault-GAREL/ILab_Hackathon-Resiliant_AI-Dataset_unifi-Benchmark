#!/usr/bin/env bash
# =============================================================================
# Resilient AI Challenge — Build + Push du dataset unifié sur Runpod
# =============================================================================
# Tout-en-un : install deps → login HF → build → push Hub.
#
# Usage :
#   1) Lance un pod Runpod (CPU-only suffit, pas besoin de GPU pour le build).
#      Recommandé : 8 vCPU / 32 GB RAM / 200 GB disque (VQAv2+COCO lourds).
#   2) Upload ce dossier `code/` + `requirements.txt`.
#   3) export HF_TOKEN=hf_xxxxx      # token avec scope repo.write
#   4) bash runpod_run.sh
#
# Variables d'env (toutes optionnelles sauf HF_TOKEN) :
#   HF_TOKEN           : token HuggingFace (obligatoire)
#   HUB_ID             : repo cible       (défaut: Mathos34400/resilient-ai-unified)
#   TIER               : tier1|tier2|all  (défaut: all)
#   STRATEGY           : uniform|sqrt|task_balanced|temperature (défaut: sqrt)
#   TEMPERATURE        : si STRATEGY=temperature (défaut: 2.0)
#   TRAIN_SIZE         : nb samples dans le train mix (défaut: 350000, v0.3)
#   MAX_PER_SOURCE     : cap par dataset source (défaut: 150000)
#   CALIBRATION_SIZE   : taille split calibration PTQ (défaut: 2048 ; 0 = skip)
#   MAX_IMAGE_DIM      : resize images côté long en JPEG (défaut: 512 ; 0 = full-res)
#   JPEG_QUALITY       : qualité JPEG des images encodées (défaut: 85)
#   OUT_DIR            : dossier local   (défaut: ./data/unified)
#   PRIVATE            : 1 = dataset privé sur le Hub (défaut: 0)
#   SKIP_INSTALL       : 1 = ne pas réinstaller pip deps (défaut: 0)
#
# Exemples :
#   # Build complet full-res sur Runpod A100 80GB / 500 GB disque :
#   HF_TOKEN=hf_xxx bash runpod_run.sh
#
#   # Build rapide (smoke test ou laptop) :
#   HF_TOKEN=hf_xxx TRAIN_SIZE=5000 MAX_PER_SOURCE=500 MAX_IMAGE_DIM=512 \
#     bash runpod_run.sh
#
#   # Privé + stratégie temperature :
#   HF_TOKEN=hf_xxx PRIVATE=1 STRATEGY=temperature TEMPERATURE=4 \
#     bash runpod_run.sh
# =============================================================================

set -euo pipefail

# --- Configuration ----------------------------------------------------------
HUB_ID="${HUB_ID:-Mathos34400/resilient-ai-unified}"
TIER="${TIER:-all}"
STRATEGY="${STRATEGY:-sqrt}"
TEMPERATURE="${TEMPERATURE:-2.0}"
TRAIN_SIZE="${TRAIN_SIZE:-350000}"
MAX_PER_SOURCE="${MAX_PER_SOURCE:-150000}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-2048}"
EVAL_CAP="${EVAL_CAP:-}"  # vide = auto (max_per_source//5). Pour équilibrer test : EVAL_CAP=5000.
OUT_DIR="${OUT_DIR:-./data/unified}"
PRIVATE="${PRIVATE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# Par défaut : resize 512px + encode JPEG — indispensable pour que 200k samples
# tiennent en RAM pendant le build. MAX_IMAGE_DIM=0 désactive le resize.
export MAX_IMAGE_DIM="${MAX_IMAGE_DIM:-512}"
export JPEG_QUALITY="${JPEG_QUALITY:-85}"

# --- Vérifications préalables -----------------------------------------------
# Script vit dans scripts/ ; on remonte à la racine du projet pour que
# requirements.txt et data/ soient résolus correctement.
cd "$(dirname "$0")/.."

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERREUR : HF_TOKEN non défini. Obtenir un token écriture sur"
    echo "         https://huggingface.co/settings/tokens puis :"
    echo "         export HF_TOKEN=hf_xxx"
    exit 1
fi

if [[ ! -f requirements.txt ]]; then
    echo "ERREUR : requirements.txt introuvable dans $(pwd)"
    exit 1
fi

echo "============================================================================"
echo "Resilient AI — build dataset unifié"
echo "============================================================================"
echo "  Hub ID          : $HUB_ID"
echo "  Tier            : $TIER"
echo "  Stratégie       : $STRATEGY${STRATEGY:+ (T=$TEMPERATURE)}"
echo "  Train size      : $TRAIN_SIZE"
echo "  Max per source  : $MAX_PER_SOURCE"
echo "  Out dir         : $OUT_DIR"
echo "  Private         : $PRIVATE"
echo "  Max image dim   : ${MAX_IMAGE_DIM:-full-res}"
echo "  JPEG quality    : ${JPEG_QUALITY}"
echo "  Eval cap        : ${EVAL_CAP:-auto}"
echo "  Calibration     : ${CALIBRATION_SIZE} samples"
echo "============================================================================"

# --- 1. Install deps --------------------------------------------------------
if [[ "$SKIP_INSTALL" != "1" ]]; then
    echo ""
    echo "[1/4] Installation des dépendances Python..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo "      OK"
else
    echo ""
    echo "[1/4] Skip install (SKIP_INSTALL=1)"
fi

# --- 2. Login HuggingFace ---------------------------------------------------
echo ""
echo "[2/4] Login HuggingFace (token en env, pas écrit sur disque)..."
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
who = api.whoami()
print(f'      → connecté en tant que {who[\"name\"]} ({who.get(\"fullname\", \"?\")})')
perms = who['auth']['accessToken']
print(f'      → scope: {perms[\"role\"]}')
"

# --- 3. Dry-run (estimation tailles + poids) --------------------------------
echo ""
echo "[3/4] Dry-run : estimation des tailles natives HF et des poids de mix..."
python3 scripts/build_dataset.py \
    --tier "$TIER" \
    --strategy "$STRATEGY" \
    --temperature "$TEMPERATURE" \
    --train_size "$TRAIN_SIZE" \
    --dry_run || {
    echo "      ! dry-run échoué — on continue quand même vers le build réel."
}

# --- 4. Build + push --------------------------------------------------------
echo ""
echo "[4/4] Build complet + push sur le Hub..."
PUSH_ARGS=()
if [[ "$PRIVATE" == "1" ]]; then
    PUSH_ARGS+=(--private)
fi

BUILD_ARGS=(
    --tier "$TIER"
    --strategy "$STRATEGY"
    --temperature "$TEMPERATURE"
    --train_size "$TRAIN_SIZE"
    --max_per_source "$MAX_PER_SOURCE"
    --calibration_size "$CALIBRATION_SIZE"
    --out_dir "$OUT_DIR"
    --hub_id "$HUB_ID"
    --push
)
if [[ -n "$EVAL_CAP" ]]; then
    BUILD_ARGS+=(--eval_cap "$EVAL_CAP")
fi

python3 scripts/build_dataset.py "${BUILD_ARGS[@]}" "${PUSH_ARGS[@]}"

echo ""
echo "============================================================================"
echo "Build + push terminés."
echo "Dataset disponible sur : https://huggingface.co/datasets/$HUB_ID"
echo "Manifest local         : $OUT_DIR/manifest.json"
echo "============================================================================"
