#!/bin/bash
# =============================================================================
# Entrypoint Docker — exécuté à chaque démarrage du container Runpod.
#
# Responsabilités :
#   1. Créer les dossiers persistants sur /workspace (Network Volume Runpod)
#   2. Symlink /app/results et /app/reports vers /workspace pour que les
#      artefacts survivent un stop/start du pod.
#
# Les variables d'env HF_HOME, LD_LIBRARY_PATH, etc. sont définies via
# ENV dans le Dockerfile, donc déjà actives dans toutes les sessions SSH.
# =============================================================================
set -e

if [ -d /workspace ]; then
    # Cache HF et dossiers de travail
    mkdir -p /workspace/hf-cache/hub \
             /workspace/results/runs \
             /workspace/reports \
             /workspace/tmp \
             /workspace/pip-cache

    # Symlinks pour que /app/results et /app/reports pointent vers /workspace
    # (les outputs des runs survivent au stop du pod).
    for d in results reports; do
        if [ ! -L "/app/${d}" ]; then
            [ -d "/app/${d}" ] && rm -rf "/app/${d}"
            ln -s "/workspace/${d}" "/app/${d}"
        fi
    done

    echo "[entrypoint] /workspace ready : HF_HOME=$HF_HOME"
else
    echo "⚠ /workspace introuvable — les artefacts seront sur le container disk."
fi

# Délègue au CMD du Dockerfile (sleep infinity par défaut)
exec "$@"
