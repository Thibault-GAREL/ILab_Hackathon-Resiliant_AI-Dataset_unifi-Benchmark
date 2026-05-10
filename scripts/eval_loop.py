#!/usr/bin/env python
"""
eval_loop.py — orchestrateur de benchmark + tracking énergie.

DÉCISION 2026-04-30 (Mathis) :
  - Plus d'évaluation OOD (MMStar / MME / POPE) en flow par défaut. Ces
    benchmarks restent disponibles dans `eval_ood.py` pour usage manuel
    ponctuel mais ne sont PLUS appelés par cet orchestrateur.
  - Le bench tourne sur **10% du dataset unifié** par défaut (subsample
    déterministe par-source, seed=42). Override : `--subsample_pct 100`
    pour passer le full split.
  - Tracking énergie CodeCarbon **systématique**, par-source, agrégé en
    fin de run, dump CSV global (registry) + artefacts détaillés JSON.
    C'est obligatoire : le ranking PEReN est dominé par l'énergie.

Usage typique :

    # Smoke test (oracle, ~1 min) :
    python eval_loop.py \
        --mode oracle \
        --dataset_local ./data/unified \
        --subsample_pct 5 \
        --run_label smoke-oracle \
        --out reports/smoke.json

    # Run réel sur un checkpoint quantizé (10% par défaut) :
    python eval_loop.py \
        --mode real \
        --model_path runs/student-qat-int4/final \
        --dataset_hub Mathos34400/resilient-ai-unified \
        --run_label qat-int4-v2 \
        --out reports/qat-int4.json

    # Full split (1h+ sur A100) :
    python eval_loop.py \
        --mode real \
        --model_path runs/student-final/final \
        --dataset_hub Mathos34400/resilient-ai-unified \
        --subsample_pct 100 \
        --run_label final-full \
        --out reports/final-full.json

Outputs générés :
  - <out>                                         : rapport JSON principal
  - results/registry.csv                          : append-only, 1 ligne / run
  - results/runs/<label>_<ts>.json                : artefact détaillé
  - results/runs/<label>_<ts>_energy.csv          : énergie par-source
  - results/runs/<label>_<ts>_summary.txt         : résumé human-readable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.benchmark import (
    evaluate_model,
    make_gemma_generate_fn,
    constant_generate_fn,
    oracle_generate_fn,
    DEFAULT_SUBSAMPLE_PCT,
    DEFAULT_SUBSAMPLE_SEED,
)
from src.tracking.energy_tracker import (
    EnergyTracker,
    aggregate_snapshots,
    derive_efficiency_metrics,
)
from src.tracking.run_registry import (
    RegistryEntry,
    append_to_registry,
    dump_run_artifacts,
)


def _load_internal_dataset(args):
    from datasets import load_from_disk, load_dataset
    if args.dataset_local:
        dd = load_from_disk(args.dataset_local)
        return dd[args.split]
    if args.dataset_hub:
        return load_dataset(args.dataset_hub, split=args.split)
    return None


def _make_generate_fn(args):
    if args.mode == "oracle":
        return oracle_generate_fn(), "oracle"
    if args.mode == "constant":
        return constant_generate_fn(args.constant_answer), f"constant:{args.constant_answer}"

    # real — route entre les backends transformers et llama-cpp-python
    if args.backend == "llama_cpp":
        if not args.gguf_repo or not args.gguf_filename:
            raise SystemExit(
                "--backend llama_cpp requiert --gguf_repo et --gguf_filename."
            )
        from src.models.llama_cpp_backend import make_llama_cpp_generate_fn
        gen_fn = make_llama_cpp_generate_fn(
            gguf_repo=args.gguf_repo,
            gguf_filename=args.gguf_filename,
            mmproj_repo=args.mmproj_repo,
            mmproj_filename=args.mmproj_filename,
            max_new_tokens=args.max_new_tokens,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            model_family=args.model_family,
        )
        model_id = f"{args.gguf_repo}/{args.gguf_filename}"
        return gen_fn, model_id

    # transformers (défaut)
    if not args.model_path:
        raise SystemExit("--model_path requis en mode real avec backend transformers.")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] {args.model_path} sur {args.device}…")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    gen_fn = make_gemma_generate_fn(
        model, processor,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    return gen_fn, args.model_path


def main() -> int:
    ap = argparse.ArgumentParser()

    # ----- Modèle ---------------------------------------------------------
    ap.add_argument("--mode", choices=["real", "oracle", "constant"], default="real")
    ap.add_argument("--backend", choices=["transformers", "llama_cpp"],
                    default="transformers",
                    help="Backend d'inférence. transformers = HF safetensors / "
                         "bnb-4bit / AWQ. llama_cpp = GGUF via llama-cpp-python.")
    ap.add_argument("--model_path", default=None,
                    help="Repo HF ou chemin local — requis si backend=transformers.")
    ap.add_argument("--constant_answer", default="unknown")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=32)

    # ----- Backend llama-cpp-python (GGUF) -------------------------------
    ap.add_argument("--gguf_repo", default=None,
                    help="Repo HF (ex: bartowski/gemma-3n-E4B-it-GGUF) "
                         "ou dossier local contenant le .gguf.")
    ap.add_argument("--gguf_filename", default=None,
                    help="Nom du fichier .gguf principal (glob accepté côté HF).")
    ap.add_argument("--mmproj_repo", default=None,
                    help="Repo HF du mmproj-*.gguf (multimodal projector).")
    ap.add_argument("--mmproj_filename", default=None,
                    help="Nom du fichier mmproj-*.gguf.")
    ap.add_argument("--model_family", default="gemma3",
                    help="Famille du modèle (gemma3 / gemma4 / llava). "
                         "Pilote le ChatHandler llama.cpp.")
    ap.add_argument("--n_gpu_layers", type=int, default=-1,
                    help="-1 = toutes les couches sur GPU. 0 = CPU only. "
                         "Override pour fit dans une VRAM limitée.")
    ap.add_argument("--n_ctx", type=int, default=2048,
                    help="Taille du contexte llama.cpp.")

    # ----- Dataset interne ------------------------------------------------
    ap.add_argument("--dataset_local", default=None,
                    help="DatasetDict local (save_to_disk).")
    ap.add_argument("--dataset_hub", default=None,
                    help="HF Hub id (ex: Mathos34400/resilient-ai-unified).")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Filtre : sous-ensemble de sources.")
    ap.add_argument("--limit_per_source", type=int, default=None,
                    help="Cap dur après subsample (smoke test rapide).")
    ap.add_argument("--subsample_pct", type=float, default=DEFAULT_SUBSAMPLE_PCT,
                    help=f"%% par-source à évaluer (défaut: {DEFAULT_SUBSAMPLE_PCT}). "
                         "100 = full split.")
    ap.add_argument("--subsample_seed", type=int, default=DEFAULT_SUBSAMPLE_SEED,
                    help=f"Seed pour le subsample (défaut: {DEFAULT_SUBSAMPLE_SEED}).")

    # ----- Tracking énergie ----------------------------------------------
    ap.add_argument("--no_energy_tracking", action="store_true",
                    help="Désactive CodeCarbon (debug uniquement).")
    ap.add_argument("--country_iso", default="FRA",
                    help="Code pays CodeCarbon (défaut: FRA, override pour Runpod US).")

    # ----- Output --------------------------------------------------------
    ap.add_argument("--out", type=Path, required=True,
                    help="Chemin du rapport JSON principal.")
    ap.add_argument("--run_label", default=None,
                    help="Label court pour identifier la run dans le registry "
                         "(défaut: dérivé du model_path / mode).")
    ap.add_argument("--registry_csv", type=Path,
                    default=Path("results/registry.csv"),
                    help="CSV append-only de toutes les runs.")
    ap.add_argument("--artifacts_dir", type=Path,
                    default=Path("results/runs"),
                    help="Dossier des artefacts détaillés par run.")
    ap.add_argument("--notes", default="",
                    help="Texte libre stocké dans le registry CSV.")

    args = ap.parse_args()

    # ----- Garde-fous cohérence -------------------------------------------
    if not args.dataset_local and not args.dataset_hub:
        raise SystemExit(
            "Le benchmark requiert --dataset_local OU --dataset_hub."
        )
    if args.subsample_pct <= 0 or args.subsample_pct > 100:
        raise SystemExit("--subsample_pct doit être dans ]0, 100].")

    # ----- Label par défaut ----------------------------------------------
    if args.run_label is None:
        if args.mode == "real" and args.model_path:
            args.run_label = Path(args.model_path).name or "real"
        elif args.mode == "oracle":
            args.run_label = "oracle"
        else:
            args.run_label = f"constant-{args.constant_answer}"

    t0 = time.time()
    print("=" * 72)
    print(f"eval_loop — label={args.run_label}  mode={args.mode}")
    print(f"           subsample={args.subsample_pct}% (seed={args.subsample_seed})  "
          f"split={args.split}")
    print("=" * 72)

    # ----- Inference fn ---------------------------------------------------
    gen_fn, model_id = _make_generate_fn(args)

    # ----- Layer 1 : benchmark interne par-source -------------------------
    print("\n" + "─" * 60)
    print("Benchmark interne par-source (avec tracking énergie)")
    print("─" * 60)
    ds = _load_internal_dataset(args)
    print(f"[data] {args.split}: {len(ds)} samples (avant subsample)")

    bench_result, per_source_snapshots = evaluate_model(
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

    # ----- Snapshot agrégé ------------------------------------------------
    if per_source_snapshots:
        agg_snap = aggregate_snapshots(
            per_source_snapshots,
            project=args.run_label,
            task="all_sources",
        )
        agg_snap.score = bench_result.macro_avg
        agg_snap.n_samples = sum(r.n_samples for r in bench_result.per_source)
    else:
        # Pas de tracking → on construit un snapshot vide pour le registry
        from energy_tracker import EnergySnapshot, hardware_fingerprint
        import uuid
        agg_snap = EnergySnapshot(
            run_id=uuid.uuid4().hex[:12],
            project=args.run_label, task="all_sources",
            started_at=t0, duration_s=time.time() - t0,
            kwh=0.0, kwh_cpu=0.0, kwh_gpu=0.0, kwh_ram=0.0, co2_g=0.0,
            country_iso=None,
            n_samples=sum(r.n_samples for r in bench_result.per_source),
            score=bench_result.macro_avg,
            hardware=hardware_fingerprint(),
            backend="disabled",
        )

    # ----- Append au registry CSV global ---------------------------------
    config_snapshot = {
        "mode": args.mode,
        "model_path": args.model_path,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "split": args.split,
        "sources_filter": args.sources,
        "limit_per_source": args.limit_per_source,
        "subsample_pct": args.subsample_pct,
        "subsample_seed": args.subsample_seed,
        "country_iso": args.country_iso,
        "no_energy_tracking": args.no_energy_tracking,
    }
    entry = RegistryEntry.from_benchmark(
        run_label=args.run_label,
        bench_result=bench_result,
        energy_snapshot=agg_snap,
        config=config_snapshot,
        notes=args.notes,
        subsample_pct=args.subsample_pct,
        subsample_seed=args.subsample_seed,
    )
    registry_path = append_to_registry(entry, csv_path=args.registry_csv)
    print(f"[registry] +1 row → {registry_path}")

    # ----- Dump artefacts détaillés --------------------------------------
    artefacts = dump_run_artifacts(
        run_label=args.run_label,
        bench_result=bench_result,
        energy_snapshot=agg_snap,
        per_source_snapshots=per_source_snapshots,
        out_dir=args.artifacts_dir,
        config=config_snapshot,
    )
    for k, p in artefacts.items():
        print(f"[artefact] {k:10s} → {p}")

    # ----- Rapport JSON principal ----------------------------------------
    summary: dict[str, Any] = {
        "macro_avg": bench_result.macro_avg,
        "micro_avg": bench_result.micro_avg,
        "per_source": {r.source: r.score for r in bench_result.per_source},
        "n_samples_total": agg_snap.n_samples,
        "n_sources": len(bench_result.per_source),
        "energy": {
            "kwh_total": agg_snap.kwh,
            "kwh_gpu": agg_snap.kwh_gpu,
            "kwh_cpu": agg_snap.kwh_cpu,
            "kwh_ram": agg_snap.kwh_ram,
            "co2_g": agg_snap.co2_g,
            "duration_s": agg_snap.duration_s,
            "backend": agg_snap.backend,
            "country": agg_snap.country_iso,
        },
        "efficiency_kpi": derive_efficiency_metrics(agg_snap),
    }
    report = {
        "model_id": model_id,
        "run_label": args.run_label,
        "timestamp": t0,
        "wall_time_s": time.time() - t0,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "results": json.loads(bench_result.to_json()),
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # ----- Console summary -----------------------------------------------
    print("\n" + "=" * 72)
    print(f"Rapport principal → {args.out}")
    print("=" * 72)
    print(f"  macro_avg       = {summary['macro_avg']:.4f}")
    print(f"  micro_avg       = {summary['micro_avg']:.4f}")
    print(f"  n_samples       = {summary['n_samples_total']}")
    print(f"  kWh total       = {summary['energy']['kwh_total']:.6g}")
    print(f"    └─ GPU        = {summary['energy']['kwh_gpu']:.6g}")
    print(f"    └─ CPU        = {summary['energy']['kwh_cpu']:.6g}")
    print(f"    └─ RAM        = {summary['energy']['kwh_ram']:.6g}")
    print(f"  CO2 (gCO2eq)    = {summary['energy']['co2_g']:.4g}")
    eff = summary["efficiency_kpi"]
    print(f"  J/sample        = {eff['joules_per_sample']:.4g}")
    print(f"  Wh/sample       = {eff['wh_per_sample']:.4g}")
    if "score_per_kwh" in eff:
        print(f"  score/kWh       = {eff['score_per_kwh']:.4g}")
    print(f"  Wall time       = {report['wall_time_s']:.1f}s")
    print(f"  Backend         = {summary['energy']['backend']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
