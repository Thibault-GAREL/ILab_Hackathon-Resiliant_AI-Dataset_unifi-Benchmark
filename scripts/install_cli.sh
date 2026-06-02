#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_SCRIPT="${PROJECT_ROOT}/scripts/bench"
PYTHON_BIN="${PYTHON:-python3}"

chmod +x "$BENCH_SCRIPT"

echo "[install_cli] bench executable: $BENCH_SCRIPT"

if ! "$PYTHON_BIN" -c "import typer, rich, questionary, yaml" >/dev/null 2>&1; then
    echo ""
    echo "Deps CLI manquantes pour $PYTHON_BIN."
    read -r -p "Installer via requirements.txt maintenant ? [y/N] " install_answer
    case "$install_answer" in
        y|Y|yes|YES)
            "$PYTHON_BIN" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
            ;;
        *)
            echo "Commande a lancer plus tard:"
            echo "  $PYTHON_BIN -m pip install -r ${PROJECT_ROOT}/requirements.txt"
            ;;
    esac
fi

if [ -d /workspace ] && [ "$PROJECT_ROOT" = "/app" ]; then
    ALIAS_LINE="alias bench=/app/scripts/bench"
    echo ""
    echo "Runpod detecte."
    echo "Alias conseille:"
    echo "  $ALIAS_LINE"
    echo ""
    read -r -p "Ajouter cet alias a ~/.bashrc ? [y/N] " answer
    case "$answer" in
        y|Y|yes|YES)
            if ! grep -Fxq "$ALIAS_LINE" "$HOME/.bashrc" 2>/dev/null; then
                printf '\n%s\n' "$ALIAS_LINE" >> "$HOME/.bashrc"
            fi
            echo "[install_cli] alias ajoute. Recharge avec: source ~/.bashrc"
            ;;
        *)
            echo "[install_cli] alias non ajoute. Commande a copier:"
            echo "  $ALIAS_LINE"
            ;;
    esac
else
    echo ""
    echo "Commande directe:"
    echo "  $BENCH_SCRIPT --help"
    echo ""
    echo "Alias optionnel:"
    echo "  alias bench=$BENCH_SCRIPT"
fi
