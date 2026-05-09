#!/usr/bin/env python
"""
Entrypoint CLI court pour le benchmark interne (subsample 10% par défaut +
tracking énergie). Pour le flow complet avec registry / artefacts, utiliser
`eval_loop.py` qui appelle ce module.

Usage :
    # Benchmark d'un checkpoint Gemma 3n/4 réel sur Runpod
    python run_benchmark.py \
        --model_path google/gemma-3n-E2B-it \
        --dataset_hub Mathos34400/resilient-ai-unified \
        --split test \
        --out results/gemma3n-e2b-baseline.json

    # Smoke test (oracle, doit score ~1.0 sauf CIDEr noisy)
    python run_benchmark.py \
        --mode oracle \
        --dataset_local ./data/unified \
        --split validation \
        --subsample_pct 5 \
        --out results/oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark import (
    evaluate_model,
    make_gemma_generate_fn,
    constant_generate_fn,
    oracle_generate_fn,
    DEFAULT_SUBSAMPLE_PCT,
    DEFAULT_SUBSAMPLE_SEED,
)


def load_dataset_split(args):
    from datasets import load_from_disk, load_dataset
    if args.dataset_local:
        dd = load_from_disk(args.dataset_local)
        return dd[args.split]
    if args.dataset_hub:
        return load_dataset(args.dataset_hub, split=args.split)
    raise SystemExit("Il faut spécifier --dataset_local OU --dataset_hub.")


def make_generate_fn(args):
    if args.mode == "oracle":
        return oracle_generate_fn(), "oracle"
    if args.mode == "constant":
        return constant_generate_fn(args.constant_answer), f"constant:{args.constant_answer}"
    if not args.model_path:
        raise SystemExit("--model_path requis en mode real.")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] {args.model_path} sur {args.device}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    return make_gemma_generate_fn(model, processor, device=args.device), args.model_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "oracle", "constant"], default="real")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--constant_answer", default="unknown")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--dataset_local", default=None)
    ap.add_argument("--dataset_hub", default=None)
    ap.add_argument("--split", default="test", choices=["validation", "test"])

    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--limit_per_source", type=int, default=None)
    ap.add_argument("--subsample_pct", type=float, default=DEFAULT_SUBSAMPLE_PCT,
                    help=f"%% par-source (défaut: {DEFAULT_SUBSAMPLE_PCT}). 100 = full.")
    ap.add_argument("--subsample_seed", type=int, default=DEFAULT_SUBSAMPLE_SEED)
    ap.add_argument("--no_energy_tracking", action="store_true")
    ap.add_argument("--run_label", default="run_benchmark")

    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = load_dataset_split(args)
    print(f"[data] {args.split}: {len(ds)} samples (avant subsample)")

    gen_fn, model_id = make_generate_fn(args)
    result, _snaps = evaluate_model(
        ds,
        gen_fn,
        model_id=model_id,
        split=args.split,
        limit_per_source=args.limit_per_source,
        sources=args.sources,
        subsample_pct=args.subsample_pct,
        subsample_seed=args.subsample_seed,
        track_energy=not args.no_energy_tracking,
        project_label=args.run_label,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.to_json())
    print("\n" + "=" * 60)
    print(f"Résultats → {args.out}")
    print(f"  micro = {result.micro_avg:.4f}")
    print(f"  macro = {result.macro_avg:.4f}")
    print(f"  wall  = {result.wall_time_s:.1f}s")
    if result.energy_total:
        e = result.energy_total
        print(f"  kWh   = {e.get('kwh', 0):.6g}  (gpu={e.get('kwh_gpu', 0):.4g})")
        print(f"  CO2g  = {e.get('co2_g', 0):.4g}")
    print("Pour le tracking complet (registry + artefacts) : "
          "utiliser `eval_loop.py`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
