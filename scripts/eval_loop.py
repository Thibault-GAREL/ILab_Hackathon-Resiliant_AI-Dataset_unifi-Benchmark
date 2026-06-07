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
from collections import defaultdict
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

DEFAULT_INTERNAL_SOURCES = {
    "vqav2",
    "docvqa",
    "textvqa",
    "mmmu",
    "chartqa",
    "ai2d",
    "realworldqa",
    "coco_captions",
    "ok_vqa",
    "scienceqa",
    "infographicvqa",
    "mathvista",
}


def _fmt_score(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _load_internal_dataset(args):
    from datasets import load_from_disk, load_dataset
    try:
        if args.dataset_local:
            dd = load_from_disk(args.dataset_local)
            return dd[args.split]
        if args.dataset_hub:
            if args.dataset_streaming_per_source:
                stream = load_dataset(args.dataset_hub, split=args.split, streaming=True)
                target = args.dataset_streaming_per_source
                wanted = set(args.sources or DEFAULT_INTERNAL_SOURCES)
                buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in stream:
                    src = row.get("source")
                    if wanted and src not in wanted:
                        continue
                    if len(buckets[src]) < target:
                        buckets[src].append(row)
                    if wanted and all(len(buckets[src]) >= target for src in wanted):
                        break
                rows = [row for src in sorted(buckets) for row in buckets[src]]
                if not rows:
                    raise SystemExit(
                        f"Aucun sample lu depuis {args.dataset_hub}/{args.split}."
                    )
                counts = ", ".join(f"{src}={len(rows)}" for src, rows in sorted(buckets.items()))
                print(f"[data] streaming hub par-source: {counts}")
                return rows
            if args.dataset_streaming_take:
                stream = load_dataset(args.dataset_hub, split=args.split, streaming=True)
                rows = []
                for i, row in enumerate(stream):
                    if i >= args.dataset_streaming_take:
                        break
                    rows.append(row)
                if not rows:
                    raise SystemExit(
                        f"Aucun sample lu depuis {args.dataset_hub}/{args.split}."
                    )
                print(f"[data] streaming hub: {len(rows)} samples chargés")
                return rows
            return load_dataset(args.dataset_hub, split=args.split)
    except Exception as exc:
        if exc.__class__.__name__ == "DatasetNotFoundError":
            raise SystemExit(
                f"Dataset Hub inaccessible: {args.dataset_hub}. "
                "Lance `hf auth login` ou exporte HF_TOKEN dans ce container."
            )
        raise
    return None


def _make_generate_fn(args):
    if args.mode == "oracle":
        return oracle_generate_fn(), "oracle"
    if args.mode == "constant":
        return constant_generate_fn(args.constant_answer), f"constant:{args.constant_answer}"

    # real — dispatch backend
    if args.backend == "vllm":
        if not args.model_path:
            raise SystemExit("--model_path requis en mode real avec backend vllm.")
        from src.models.vllm_backend import make_vllm_generate_fn
        gen_fn = make_vllm_generate_fn(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_num_seqs=args.vllm_max_num_seqs,
            dtype=args.vllm_dtype,
            quantization=args.vllm_quantization,
            kv_cache_dtype=args.vllm_kv_cache_dtype,
            max_soft_tokens=args.vllm_max_soft_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enforce_eager=args.vllm_enforce_eager,
            revision=args.revision,
            seed=args.vllm_seed,
            adaptive_soft_tokens=args.vllm_adaptive_soft_tokens,
            adaptive_throttled_soft_tokens=args.vllm_adaptive_throttled_soft_tokens,
            adaptive_peak_power_w=args.vllm_adaptive_peak_power_w,
            adaptive_peak_active_power_w=args.vllm_adaptive_peak_active_power_w,
            adaptive_baseline_power_w=args.vllm_adaptive_baseline_power_w,
            adaptive_cooldown_requests=args.vllm_adaptive_cooldown_requests,
            adaptive_gpu_index=args.vllm_adaptive_gpu_index,
            adaptive_sample_interval_s=args.vllm_adaptive_sample_interval_s,
        )
        return gen_fn, args.model_path

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
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        model_id = f"{args.gguf_repo}/{args.gguf_filename}"
        return gen_fn, model_id

    # transformers (défaut)
    if not args.model_path:
        raise SystemExit("--model_path requis en mode real avec backend transformers.")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] {args.model_path} sur {args.device}…")

    # Workaround pour les modèles bitsandbytes 4-bit : passer device_map='cuda'
    # à from_pretrained() casse l'init du FP4 quantization state sur Gemma 4
    # (AssertionError bitsandbytes:415, FP4 state not initialized). Solution :
    # charger sur CPU puis .to(cuda) après — les couches FP4 sont alors
    # initialisées correctement par le .to() qui appelle .cuda() en interne.
    is_4bit = ('4bit' in args.model_path.lower()) or ('bnb' in args.model_path.lower())
    load_kwargs = {"torch_dtype": torch.bfloat16}
    if not is_4bit:
        load_kwargs["device_map"] = args.device

    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **load_kwargs)
    if is_4bit:
        model = model.to(args.device)
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
    ap.add_argument("--backend", choices=["transformers", "llama_cpp", "vllm"],
                    default="transformers",
                    help="Backend d'inférence. transformers = HF safetensors / "
                         "bnb-4bit / AWQ. llama_cpp = GGUF via llama-cpp-python. "
                         "vllm = serveur/offline vLLM pour safetensors compatibles.")
    ap.add_argument("--model_path", default=None,
                    help="Repo HF ou chemin local — requis si backend=transformers.")
    ap.add_argument("--constant_answer", default="unknown")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature, surtout utilisé par vLLM.")
    ap.add_argument("--top_p", type=float, default=0.95,
                    help="top_p sampling, surtout utilisé par vLLM.")
    ap.add_argument("--top_k", type=int, default=64,
                    help="top_k sampling, surtout utilisé par vLLM.")
    ap.add_argument("--revision", default=None,
                    help="Révision HF optionnelle pour les backends qui la supportent.")

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

    # ----- Backend vLLM ---------------------------------------------------
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_max_model_len", type=int, default=2048)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_num_seqs", type=int, default=1)
    ap.add_argument("--vllm_dtype", default="auto")
    ap.add_argument("--vllm_quantization", default=None,
                    help="Ex: fp8, modelopt, awq selon support vLLM.")
    ap.add_argument("--vllm_kv_cache_dtype", default=None,
                    help="Ex: fp8 pour KV cache quantifié.")
    ap.add_argument("--vllm_max_soft_tokens", type=int, default=None,
                    help="Gemma4 image processor max_soft_tokens via mm_processor_kwargs.")
    ap.add_argument("--vllm_enforce_eager", action="store_true", default=False)
    ap.add_argument("--vllm_no_enforce_eager", dest="vllm_enforce_eager", action="store_false")
    ap.add_argument("--vllm_seed", type=int, default=42,
                    help="Seed sampling vLLM quand supporté par la version installée.")
    ap.add_argument("--vllm_adaptive_soft_tokens",
                    choices=["none", "nvml_peak_v1"], default="none",
                    help="Throttle dynamique du budget visuel vLLM après pic NVML.")
    ap.add_argument("--vllm_adaptive_throttled_soft_tokens", type=int, default=70)
    ap.add_argument("--vllm_adaptive_peak_power_w", type=float, default=None)
    ap.add_argument("--vllm_adaptive_peak_active_power_w", type=float, default=None)
    ap.add_argument("--vllm_adaptive_baseline_power_w", type=float, default=0.0)
    ap.add_argument("--vllm_adaptive_cooldown_requests", type=int, default=3)
    ap.add_argument("--vllm_adaptive_gpu_index", type=int, default=0)
    ap.add_argument("--vllm_adaptive_sample_interval_s", type=float, default=0.2)

    # ----- Dataset interne ------------------------------------------------
    ap.add_argument("--dataset_local", default=None,
                    help="DatasetDict local (save_to_disk).")
    ap.add_argument("--dataset_hub", default=None,
                    help="HF Hub id (ex: Mathos34400/resilient-ai-unified).")
    ap.add_argument("--dataset_streaming_take", type=int, default=None,
                    help="Charge seulement les N premiers samples du hub en streaming "
                         "(utile pour smoke Runpod sans télécharger tout le dataset).")
    ap.add_argument("--dataset_streaming_per_source", type=int, default=None,
                    help="Charge N samples par source en streaming depuis le hub "
                         "(utile pour reproduire limit_per_source sans full download).")
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

    # ----- Dataset --------------------------------------------------------
    print("\n" + "─" * 60)
    print("Benchmark interne par-source (avec tracking énergie)")
    print("─" * 60)
    ds = _load_internal_dataset(args)
    print(f"[data] {args.split}: {len(ds)} samples (avant subsample)")

    # ----- Inference fn ---------------------------------------------------
    gen_fn, model_id = _make_generate_fn(args)

    # ----- Layer 1 : benchmark interne par-source -------------------------
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
        country_iso=args.country_iso,
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
        from src.tracking.energy_tracker import EnergySnapshot, hardware_fingerprint
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

    runtime_stats: dict[str, Any] = {}
    get_runtime_stats = getattr(gen_fn, "runtime_stats", None)
    if callable(get_runtime_stats):
        try:
            runtime_stats = get_runtime_stats()
        except Exception as exc:
            runtime_stats = {"error": f"{type(exc).__name__}: {exc}"}

    # ----- Append au registry CSV global ---------------------------------
    config_snapshot = {
        "mode": args.mode,
        "model_path": args.model_path,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "backend": args.backend,
        "revision": args.revision,
        "vllm": {
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "max_model_len": args.vllm_max_model_len,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_num_seqs": args.vllm_max_num_seqs,
            "dtype": args.vllm_dtype,
            "quantization": args.vllm_quantization,
            "kv_cache_dtype": args.vllm_kv_cache_dtype,
            "max_soft_tokens": args.vllm_max_soft_tokens,
            "enforce_eager": args.vllm_enforce_eager,
            "seed": args.vllm_seed,
            "adaptive_soft_tokens": args.vllm_adaptive_soft_tokens,
            "adaptive_throttled_soft_tokens": args.vllm_adaptive_throttled_soft_tokens,
            "adaptive_peak_power_w": args.vllm_adaptive_peak_power_w,
            "adaptive_peak_active_power_w": args.vllm_adaptive_peak_active_power_w,
            "adaptive_baseline_power_w": args.vllm_adaptive_baseline_power_w,
            "adaptive_cooldown_requests": args.vllm_adaptive_cooldown_requests,
            "adaptive_gpu_index": args.vllm_adaptive_gpu_index,
            "adaptive_sample_interval_s": args.vllm_adaptive_sample_interval_s,
        },
        "split": args.split,
        "sources_filter": args.sources,
        "limit_per_source": args.limit_per_source,
        "subsample_pct": args.subsample_pct,
        "subsample_seed": args.subsample_seed,
        "country_iso": args.country_iso,
        "no_energy_tracking": args.no_energy_tracking,
        "runtime_stats": runtime_stats,
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
        "official_like": (
            asdict(bench_result.official_like)
            if bench_result.official_like is not None else None
        ),
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
        "runtime_stats": runtime_stats,
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
    if summary.get("official_like"):
        official = summary["official_like"]
        print(f"  document        = {_fmt_score(official['document_analysis'])}")
        print(f"  image           = {_fmt_score(official['image_understanding'])}")
        print(f"  official mean   = {_fmt_score(official['mean'])}")
        print(f"  recovery        = {_fmt_score(official['mean_recovery'])}")
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
    cleanup = getattr(gen_fn, "close", None)
    if args.backend == "vllm":
        # vLLM 0.19 peut crasher dans shutdown/finalizers après un run réussi.
        # On nettoie d'abord les enfants EngineCore pour que bench_all puisse
        # enchaîner plusieurs variantes sur une seule L4.
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        import os
        import subprocess
        subprocess.run(
            ["pkill", "-P", str(os.getpid())],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os._exit(0)
    if callable(cleanup):
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
