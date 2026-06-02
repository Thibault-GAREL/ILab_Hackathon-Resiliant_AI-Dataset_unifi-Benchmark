#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[vllm] serving modele-complet-fp8-kvfp8-soft140 on :8000"
vllm serve --config vllm_config.yaml &
pid_a=$!

echo "[vllm] serving modele-complet-lora-energie-soft140 on :8001"
vllm serve --config vllm_config_lora_energie.yaml &
pid_b=$!

trap 'kill "$pid_a" "$pid_b" 2>/dev/null || true' INT TERM EXIT
wait
