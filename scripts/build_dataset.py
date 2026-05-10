#!/usr/bin/env python
"""
Script de construction + push du dataset unifié sur HuggingFace Hub.

Usage typique (sur Runpod, après `pip install -r requirements.txt`) :

    python build_dataset.py \
        --tier all \
        --strategy sqrt \
        --train_size 200000 \
        --max_per_source 150000 \
        --hub_id mathis-lacombe/resilient-ai-unified \
        --push

Ou en dry-run local (sans push, pour inspection des tailles/poids) :

    python build_dataset.py --tier tier1 --strategy sqrt --dry_run

Le script :
1. charge les datasets sources via loaders.py,
2. applique les splits (officiels quand dispo, sinon 80/10/10 interne, cf. mix.py),
3. compose le train set unifié selon la stratégie choisie,
4. garde val/test **par-source** (pas de mix) pour éval comparable,
5. construit 3 `datasets.Dataset` (train / val / test) via le schema canonique,
6. sérialise en local + (optionnel) pousse sur le Hub.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ALL_DATASETS,
    DEFAULT_HUB_REPO,
    DATASET_VERSION,
    MixConfig,
    SplitConfig,
    TIER1,
    TIER2,
)
from src.data.mix import (
    build_source_splits,
    compose_train,
    compose_eval,
    compose_calibration,
    compute_weights,
    describe_mix,
)
from src.data.schema import UnifiedSample, hf_features


def _samples_to_hf_dict(samples: list[UnifiedSample]) -> list[dict]:
    out = []
    for s in samples:
        d = s.to_dict()
        d["metadata"] = json.dumps(d.get("metadata") or {}, ensure_ascii=False)
        # datasets attend choices/correct_choice_idx non-None ; on normalise
        d["choices"] = d["choices"] or []
        if d["correct_choice_idx"] is None:
            d["correct_choice_idx"] = -1
        out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["tier1", "tier2", "all"], default="all")
    ap.add_argument(
        "--strategy",
        choices=["uniform", "sqrt", "task_balanced", "temperature"],
        default="sqrt",
    )
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--train_size", type=int, default=200_000,
                    help="Nombre total de samples dans le train mix final.")
    ap.add_argument("--max_per_source", type=int, default=None,
                    help="Cap par dataset source avant mix (évite RAM sur VQAv2).")
    ap.add_argument("--eval_cap", type=int, default=None,
                    help="Cap par source sur val/test (défaut: max_per_source//5). "
                         "Utile pour équilibrer le benchmark quand une source (COCO test) domine.")
    ap.add_argument("--calibration_size", type=int, default=2048,
                    help="Taille du split calibration pour quantization PTQ "
                         "(AWQ/GPTQ). 0 = pas de calibration split. Défaut: 2048.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("./data/unified"))
    ap.add_argument("--hub_id", default=DEFAULT_HUB_REPO)
    ap.add_argument("--push", action="store_true", help="Pousse sur HuggingFace Hub.")
    ap.add_argument("--private", action="store_true", help="Dataset Hub privé.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Affiche le mix sans charger les données.")
    args = ap.parse_args()

    if args.tier == "tier1":
        selected = TIER1
    elif args.tier == "tier2":
        selected = TIER2
    else:
        selected = TIER1 + TIER2

    mix_cfg = MixConfig(
        strategy=args.strategy,
        temperature=args.temperature,
        max_per_dataset=args.max_per_source,
        seed=args.seed,
    )
    split_cfg = SplitConfig(use_official_splits=True, seed=args.seed)

    print("=" * 72)
    print(f"Resilient AI — build dataset unifié ({DATASET_VERSION})")
    print(f"Tier      : {args.tier}  ({len(selected)} datasets)")
    print(f"Stratégie : {mix_cfg.strategy}" + (f" (T={mix_cfg.temperature})" if mix_cfg.strategy == "temperature" else ""))
    print(f"Seed      : {mix_cfg.seed}")
    print("=" * 72)

    # ----- DRY RUN : ne charge pas les images, juste les tailles ------------
    if args.dry_run:
        print("\n[dry-run] estimation des tailles via metadata HuggingFace...")
        from datasets import load_dataset_builder
        sizes = {}
        for name in selected:
            spec = ALL_DATASETS[name]
            native = spec.splits.get("train")
            if not native:
                sizes[name] = 0
                continue
            try:
                b = load_dataset_builder(spec.hf_id, spec.config)
                info = b.info.splits.get(native)
                sizes[name] = int(info.num_examples) if info else 0
            except Exception as e:
                print(f"  ! {name}: impossible d'estimer ({e})")
                sizes[name] = 0
        print("\n" + describe_mix(sizes, mix_cfg, n_total=args.train_size))
        return 0

    # ----- Chargement + splits ----------------------------------------------
    print("\n[1/4] Chargement des sources + splits (peut être long sur VQAv2)...")
    source_splits = build_source_splits(
        selected,
        split_cfg,
        max_per_split=args.max_per_source,
        eval_cap=args.eval_cap,
    )
    sizes = {name: len(sp["train"]) for name, sp in source_splits.items()}
    print("\nTailles train après chargement :")
    print(describe_mix(sizes, mix_cfg, n_total=args.train_size))

    # ----- Split calibration AVANT train mix --------------------------------
    # IMPORTANT : on extrait la calibration d'abord, avec remove_from_pools=True.
    # Ça retire les samples piochés des pools train de source_splits, garantit
    # la disjonction stricte calibration ⟂ train (cf. discipline PTQ — pas de
    # fuite entre échelles calibrées et données de fine-tuning).
    calibration_samples: list = []
    if args.calibration_size > 0:
        print(f"\n[2a/4] Composition du split calibration "
              f"({args.calibration_size} samples visés)...")
        calibration_samples = compose_calibration(
            source_splits,
            n_total=args.calibration_size,
            seed=args.seed,
            remove_from_pools=True,
        )

    # ----- Composition train mix ---------------------------------------------
    print(f"\n[2b/4] Composition du train mix ({args.train_size} samples)...")
    train_mix = compose_train(source_splits, mix_cfg, args.train_size)
    val_by_source = compose_eval(source_splits, "val")
    test_by_source = compose_eval(source_splits, "test")

    # ----- LIBÉRATION RAM avant [3/4] ---------------------------------------
    # À ce stade, train_mix / val_by_source / test_by_source gardent leurs refs
    # sur les samples sélectionnés. On peut vider `source_splits` pour que les
    # samples NON sélectionnés (~45% du pool train en sqrt-balanced) soient GC.
    # Récupère typiquement 30-45 GB, indispensable en 1024px/q=95.
    import gc as _gc
    import json as _json
    for _name in list(source_splits.keys()):
        source_splits[_name]["train"] = []
        source_splits[_name]["val"] = []
        source_splits[_name]["test"] = []
    _gc.collect()
    print("  [mem] source_splits vidés, gc.collect() OK")

    # ----- Construction datasets HF (streaming via from_generator) -----------
    # Pourquoi from_generator et pas from_list :
    # `Dataset.from_list` force la matérialisation d'une pylist + copie Arrow
    # simultanément (2× bytes en RAM). À 1024px/q=95 × 350k samples, ça fait
    # ~170 GB de peak et ça OOM sur 128 GB.
    # `Dataset.from_generator` écrit par batchs (writer_batch_size) vers le
    # cache disque et ne garde que le batch courant en RAM. On libère aussi
    # chaque sample (samples[i]=None) AU FUR ET À MESURE du yield → la liste
    # d'origine se vide en place, permettant au GC de récupérer les bytes.
    #
    # Budget mémoire visé pour 128 GB RAM :
    #  - AVANT stream train (le plus gros) : on a déjà écrit et libéré
    #    calib+val+test, donc seul train_mix (~87 GB bytes) reste vivant.
    #  - PENDANT stream train : train_mix se vide, Arrow buffer ~1 GB.
    #  - Pic ≈ 87 + 5 = 92 GB. Marge ~36 GB.
    #
    # ORDRE CRITIQUE : calib → val → test → train (train écrit en dernier,
    # quand max de RAM est libre). Ne PAS remettre train en premier.
    print("\n[3/4] Construction des Dataset HuggingFace (streaming)...")
    from datasets import Dataset, DatasetDict

    features = hf_features()

    def _mem_gb() -> str:
        """Retourne l'usage RAM du process en GB (via /proc/self/status)."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return f"{kb / 1024 / 1024:.1f} GB"
        except Exception:
            pass
        return "?"

    def _stream_release(samples_list):
        """Yield HF-dicts, release refs au fil de l'eau → GC incrémental."""
        for i in range(len(samples_list)):
            s = samples_list[i]
            samples_list[i] = None  # libère le ref dans la liste
            yield {
                "id": s.id,
                "source": s.source,
                "task": s.task,
                "image": {"bytes": s.image, "path": None} if s.image else None,
                "question": s.question,
                "answers": list(s.answers),
                "choices": list(s.choices) if s.choices else [],
                "correct_choice_idx": (
                    s.correct_choice_idx
                    if s.correct_choice_idx is not None else -1
                ),
                "metric_hint": s.metric_hint,
                "metadata": _json.dumps(s.metadata or {}, ensure_ascii=False),
            }
            # GC agressif pendant le stream (500 samples ~ 125 MB à libérer)
            if (i + 1) % 500 == 0:
                _gc.collect()

    print(f"  [mem] avant streams : RSS={_mem_gb()}")

    # --- Capturer les tailles pour le manifest AVANT de libérer les dicts ---
    val_sizes_by_source = {k: len(v) for k, v in val_by_source.items()}
    test_sizes_by_source = {k: len(v) for k, v in test_by_source.items()}

    # ORDRE : petits d'abord pour libérer un max de RAM avant train.

    # 1. Calibration (tiny : 2k samples ≈ 0.5 GB)
    calib_ds = None
    if calibration_samples:
        calib_copy = list(calibration_samples)
        calib_ds = Dataset.from_generator(
            lambda: _stream_release(calib_copy),
            features=features,
            writer_batch_size=500,
        )
        del calib_copy
        _gc.collect()
        print(f"  calib: {len(calib_ds)}  [mem] RSS={_mem_gb()}")

    # 2. Validation (~15 GB)
    flat_val = [s for lst in val_by_source.values() for s in lst]
    del val_by_source
    _gc.collect()
    val_ds = Dataset.from_generator(
        lambda: _stream_release(flat_val),
        features=features,
        writer_batch_size=500,
    )
    del flat_val
    _gc.collect()
    print(f"  val:   {len(val_ds)}  [mem] RSS={_mem_gb()}")

    # 3. Test (~15 GB)
    flat_test = [s for lst in test_by_source.values() for s in lst]
    del test_by_source
    _gc.collect()
    test_ds = Dataset.from_generator(
        lambda: _stream_release(flat_test),
        features=features,
        writer_batch_size=500,
    )
    del flat_test
    _gc.collect()
    print(f"  test:  {len(test_ds)}  [mem] RSS={_mem_gb()}")

    # 4. Train (le plus gros, ~87 GB) — écrit EN DERNIER, quand calib/val/test
    # sont déjà écrits sur disque → max de RAM libre.
    train_ds = Dataset.from_generator(
        lambda: _stream_release(train_mix),
        features=features,
        writer_batch_size=500,
    )
    del train_mix
    _gc.collect()
    print(f"  train: {len(train_ds)}  [mem] RSS={_mem_gb()}")

    dd_splits = {"train": train_ds, "validation": val_ds, "test": test_ds}
    if calib_ds is not None:
        dd_splits["calibration"] = calib_ds
    dd = DatasetDict(dd_splits)
    calib_n = len(dd_splits.get("calibration", [])) if "calibration" in dd_splits else 0
    print(
        f"  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}"
        + (f"  calibration: {calib_n}" if calib_n else "")
    )

    # ----- Sauvegarde locale -------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.out_dir))
    # on écrit aussi le manifest (pour audit équipe)
    import os as _os
    manifest = {
        "version": DATASET_VERSION,
        "tier": args.tier,
        "strategy": args.strategy,
        "temperature": args.temperature if args.strategy == "temperature" else None,
        "train_size": len(train_ds),
        "max_per_source": args.max_per_source,
        "eval_cap": args.eval_cap,
        "max_image_dim": _os.environ.get("MAX_IMAGE_DIM", "512"),
        "jpeg_quality": _os.environ.get("JPEG_QUALITY", "85"),
        "seed": args.seed,
        "sizes_available_train": sizes,
        "weights_used": compute_weights(sizes, mix_cfg),
        "val_sizes_by_source": val_sizes_by_source,
        "test_sizes_by_source": test_sizes_by_source,
        "calibration_size": len(calibration_samples),
        "calibration_sources": sorted({s.source for s in calibration_samples}),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[4/4] Sauvegardé dans {args.out_dir}")

    # ----- Push Hub ----------------------------------------------------------
    if args.push:
        print(f"\n[push] Upload vers {args.hub_id} ...")
        dd.push_to_hub(
            args.hub_id,
            private=args.private,
            commit_message=f"Unified dataset {DATASET_VERSION} ({args.strategy})",
        )
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(args.out_dir / "manifest.json"),
            path_in_repo="manifest.json",
            repo_id=args.hub_id,
            repo_type="dataset",
            commit_message="Add build manifest",
        )
        print("  OK — repo poussé.")
    else:
        print("\n(Sans --push : n'a pas été uploadé sur le Hub.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
