#!/bin/bash
set -e

DATASET="ECE-ILAB/resilient-ai-unified"
SUBSAMPLE=10  # 10% pour itération, passer à 100 avant soumission

# Tes modèles HF (id complet sur le Hub)
declare -A MODELS=(
    ["gemma3n-baseline"]="google/gemma-3n-E4B-it"
    ["mon-qat-int4"]="ton-username/gemma3n-qat-int4"
    ["mon-pruning-50"]="ton-username/gemma3n-pruning-50pct"
    # ... ajoute autant que tu veux
)

for label in "${!MODELS[@]}"; do
    echo "═══ Bench: $label (${MODELS[$label]}) ═══"
    python eval_loop.py \
        --mode real \
        --model_path "${MODELS[$label]}" \
        --dataset_hub "$DATASET" \
        --subsample_pct "$SUBSAMPLE" \
        --run_label "$label" \
        --country_iso USA \
        --notes "HF: ${MODELS[$label]}" \
        --out "reports/${label}.json"
done

echo "═══ Done. Registry: ═══"
column -t -s, results/registry.csv | head -20

#chmod +x bench_all.sh
# ./bench_all.sh