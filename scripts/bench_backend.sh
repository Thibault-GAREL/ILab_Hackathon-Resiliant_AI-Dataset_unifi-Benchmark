#!/bin/bash
set -u

BACKEND="${1:-}"
if [ -z "$BACKEND" ]; then
    echo "Usage: bash scripts/bench_backend.sh vllm|llama_cpp [bench_all args]"
    echo "Env: LABEL=... LIMIT=2 DATASET_LOCAL=... COUNTRY=CZE"
    exit 2
fi
shift || true

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
COUNTRY="${COUNTRY:-CZE}"
LIMIT="${LIMIT:-2}"
SUBSAMPLE_PCT="${SUBSAMPLE_PCT:-100}"

if [ -z "${DATASET_LOCAL:-}" ]; then
    if [ -d /workspace/code/data/unified ]; then
        DATASET_LOCAL=/workspace/code/data/unified
    elif [ -d /app/data/unified ]; then
        DATASET_LOCAL=/app/data/unified
    else
        DATASET_LOCAL=/workspace/code/data/unified
    fi
fi

if [ -d /workspace ]; then
    export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/hf-cache/hub}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache}"
    export TMPDIR="${TMPDIR:-/workspace/tmp}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
    mkdir -p "$HF_HUB_CACHE" "$TMPDIR" "$PIP_CACHE_DIR" /workspace/results/runs /workspace/reports
    for d in results reports; do
        local_path="${PROJECT_ROOT}/${d}"
        if [ ! -L "$local_path" ]; then
            [ -d "$local_path" ] && rm -rf "$local_path"
            ln -s "/workspace/${d}" "$local_path"
        fi
    done
fi

CMD=(
    "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/bench_all.py"
    --dataset_local "$DATASET_LOCAL"
    --subsample_pct "$SUBSAMPLE_PCT"
    --country_iso "$COUNTRY"
)

case "$BACKEND" in
    vllm)
        if [ -n "${LABEL:-}" ]; then
            CMD+=(--label "$LABEL")
        else
            CMD+=(--group vllm_smoke)
        fi
        ;;
    llama_cpp)
        CMD+=(--label "${LABEL:-gemma4-e4b-gguf-q4km}")
        ;;
    *)
        echo "Backend inconnu: $BACKEND"
        echo "Choix: vllm, llama_cpp"
        exit 2
        ;;
esac

if [ -n "$LIMIT" ]; then
    CMD+=(--limit_per_source "$LIMIT")
fi

CMD+=("$@")

echo "[bench_backend] backend=$BACKEND"
echo "[bench_backend] dataset=$DATASET_LOCAL"
echo "[bench_backend] cmd: ${CMD[*]}"
cd "$PROJECT_ROOT"
"${CMD[@]}"
