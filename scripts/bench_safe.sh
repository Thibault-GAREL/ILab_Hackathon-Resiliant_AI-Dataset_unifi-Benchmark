#!/bin/bash
# =============================================================================
# Batch benchmark — robuste pour Runpod :
#   - utilise le dataset LOCAL (/workspace/code/data/unified)
#   - safety check sur la quota disque /workspace avant chaque modèle
#   - skip les modèles déjà benchmarkés (presence de reports/<label>.json)
#   - cleanup IMMÉDIAT du cache HF du modèle après chaque run pour ne pas
#     remplir le Network Volume
#   - continue même si un modèle échoue (au download ou au runtime)
#
# Pré-requis avant le 1er lancement :
#   - export HF_HOME=/workspace/hf-cache  (déjà fait via ~/.bashrc)
#   - hf auth login  (token HF avec licences Gemma 4 acceptées)
#
# Usage :
#   bash scripts/bench_safe.sh                     # lance les 8 modèles
#   nohup bash scripts/bench_safe.sh > /workspace/bench.log 2>&1 & disown
#
# Pour relancer un modèle déjà fait : supprime reports/<label>.json d'abord.
# =============================================================================
set -u

DATASET_LOCAL=${DATASET_LOCAL:-/workspace/code/data/unified}
COUNTRY=${COUNTRY:-CZE}
LIMIT=${LIMIT:-20}            # samples par source ; 20 = ~240 samples = ~5-10 min/modèle
SUBSAMPLE_PCT=${SUBSAMPLE_PCT:-100}
MIN_FREE_GB=${MIN_FREE_GB:-20}

# Format "model_path:label" — édite cette liste selon besoin.
MODELS=(
    "google/gemma-4-E4B-it:gemma4-e4b-bf16"
    "google/gemma-4-E2B-it:gemma4-e2b-bf16"
    "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit:gemma4-e4b-bnb4-dyn"
    "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit:gemma4-e2b-bnb4-dyn"
    "Chunity/gemma-4-E4B-it-AWQ-4bit:gemma4-e4b-awq-chunity"
    "cyankiwi/gemma-4-E4B-it-AWQ-INT4:gemma4-e4b-awq-cyankiwi-int4"
    "cyankiwi/gemma-4-E4B-it-AWQ-INT8:gemma4-e4b-awq-cyankiwi-int8"
    "Vishva007/gemma-4-E4B-it-W4A16-AutoRound-AWQ:gemma4-e4b-awq-autoround"
)

cd "$(dirname "$0")/.."   # racine du projet (parent de scripts/)

mkdir -p reports results/runs

for entry in "${MODELS[@]}"; do
    MODEL_PATH="${entry%%:*}"
    LABEL="${entry##*:}"

    AVAIL_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAIL_GB" -lt "$MIN_FREE_GB" ]; then
        echo "⚠ STOP : seulement ${AVAIL_GB}G libres sur /workspace (<${MIN_FREE_GB}G)."
        echo "  Augmente le Network Volume Runpod ou supprime des modèles."
        exit 1
    fi

    echo ""
    echo "═══ Bench: $LABEL ═══ (libre /workspace: ${AVAIL_GB}G)"

    if [ -f "reports/${LABEL}.json" ]; then
        echo "  SKIP — reports/${LABEL}.json existe déjà"
        continue
    fi

    if ! hf download "$MODEL_PATH"; then
        echo "  ✗ Download échoué — modèle privé/gated/non-existant. Skip."
        continue
    fi

    python scripts/eval_loop.py \
        --mode real \
        --model_path "$MODEL_PATH" \
        --dataset_local "$DATASET_LOCAL" \
        --split test \
        --subsample_pct "$SUBSAMPLE_PCT" \
        --limit_per_source "$LIMIT" \
        --country_iso "$COUNTRY" \
        --run_label "$LABEL" \
        --notes "Batch RTX 4090 limit=$LIMIT" \
        --out "reports/${LABEL}.json" \
        || echo "  ✗ Bench échoué — on continue avec le suivant"

    # Cleanup cache HF du modèle qu'on vient de tester (libère 5-15 GB).
    MODEL_DIR=$(echo "$MODEL_PATH" | sed 's|/|--|g')
    rm -rf "/workspace/hf-cache/hub/models--${MODEL_DIR}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " DONE — Registry final"
echo "═══════════════════════════════════════════════════════════════════════"
python <<'PY'
import csv
from pathlib import Path
path = Path('results/registry.csv')
if not path.exists():
    print("Pas encore de registry.")
    raise SystemExit
with path.open() as f:
    rows = list(csv.DictReader(f))
cols = ['run_label', 'macro_avg', 'kwh_total', 'samples_per_joule', 'wall_time_s']
hdr = ' | '.join(f'{c:>22}' for c in cols)
print(hdr)
print('-' * len(hdr))
for r in rows:
    print(' | '.join(f'{str(r.get(c, ""))[:22]:>22}' for c in cols))
PY
