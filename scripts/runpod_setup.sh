#!/usr/bin/env bash
# =============================================================================
# Setup d'un pod Runpod pour le benchmark Resilient AI.
#
# À lancer UNE FOIS au démarrage du pod, après clone du repo :
#
#   git clone <repo-url> && cd ILab_Resilient_AI_*
#   bash scripts/runpod_setup.sh
#
# Ce que fait le script :
#   1. Installe les deps de base (requirements.txt)
#   2. Installe les deps optionnelles pour les modèles quantifiés
#      (bitsandbytes, autoawq) — utilisé par les *-bnb-4bit / *-AWQ
#   3. Login HuggingFace (interactif, demande le token)
#   4. Crée les dossiers reports/ et results/
#   5. Smoke test rapide pour vérifier que tout marche
#
# Variables d'env optionnelles :
#   HF_TOKEN     : token HF (sinon login interactif)
#   SKIP_AWQ     : 1 = ne pas installer autoawq (pas utile si tu ne testes
#                  pas de modèle AWQ)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════════════"
echo " Setup Runpod — Resilient AI benchmark"
echo "════════════════════════════════════════════════════════════════════"

# --- 1. Deps de base --------------------------------------------------------
echo ""
echo "[1/5] Installation requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# --- 2. Deps optionnelles pour modèles quantifiés ---------------------------
echo ""
echo "[2/5] Installation deps optionnelles pour modèles quantifiés..."
echo "      bitsandbytes (NF4)..."
pip install --quiet "bitsandbytes>=0.43.0"

if [[ "${SKIP_AWQ:-0}" != "1" ]]; then
    echo "      autoawq (AWQ 4-bit)..."
    pip install --quiet autoawq || {
        echo "      ! autoawq install failed (CUDA mismatch ?). On continue sans."
    }
fi

# --- 3. Login HuggingFace ---------------------------------------------------
echo ""
echo "[3/5] Login HuggingFace..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    huggingface-cli login
fi

# --- 4. Dossiers de sortie --------------------------------------------------
echo ""
echo "[4/5] Création des dossiers reports/ et results/..."
mkdir -p reports results/runs

# --- 5. Smoke test ----------------------------------------------------------
echo ""
echo "[5/5] Smoke test (oracle, ~30 sec)..."
python scripts/eval_loop.py \
    --mode oracle \
    --dataset_hub ECE-ILAB/resilient-ai-unified \
    --subsample_pct 1 \
    --limit_per_source 5 \
    --run_label smoke-setup \
    --no_energy_tracking \
    --out reports/smoke-setup.json

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " Setup OK. Prochaines étapes :"
echo ""
echo "   # Lister les modèles disponibles :"
echo "   python scripts/bench_all.py --list"
echo ""
echo "   # Lancer un seul modèle :"
echo "   python scripts/bench_all.py --label gemma4-e4b-bf16"
echo ""
echo "   # Lancer tous les modèles :"
echo "   python scripts/bench_all.py --all"
echo ""
echo "════════════════════════════════════════════════════════════════════"
