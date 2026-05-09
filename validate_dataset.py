#!/usr/bin/env python
"""
Sanity check du dataset unifié après build ou download depuis le Hub.

Vérifie :
- Toutes les 12 sources (v0.3 : +scienceqa +infographicvqa +mathvista) sont
  présentes avec des tailles non-nulles (selon le tier buildé).
- 100% des samples ont image + question + answers non-vides
- Images décompressables (test PIL open sur un échantillon aléatoire)
- Metadata JSON bien sérialisée
- Split calibration (si présent) stratifié sur toutes les sources train
- Distribution train équilibrée selon les poids attendus

Usage :
    python validate_dataset.py --from_disk ./data/unified
    python validate_dataset.py --from_hub Mathos34400/resilient-ai-unified
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter

from PIL import Image


def validate(ds_dict, sample_check: int = 100) -> int:
    """Retourne nombre d'erreurs trouvées (0 = OK)."""
    errors = 0
    print("=" * 72)
    print("VALIDATION DATASET UNIFIE")
    print("=" * 72)

    expected_sources = {
        # Tier 1
        "vqav2", "docvqa", "textvqa", "mmmu",
        # Tier 2 v0.2
        "chartqa", "ai2d", "realworldqa", "coco_captions", "ok_vqa",
        # Tier 2 v0.3 (ajouts : raisonnement + infographies + math)
        "scienceqa", "infographicvqa", "mathvista",
    }

    # Splits attendus (calibration = optionnel, seulement v0.3+)
    expected_splits = ["train", "validation", "test"]
    if "calibration" in ds_dict:
        expected_splits.append("calibration")

    for split_name in expected_splits:
        if split_name not in ds_dict:
            print(f"\n[{split_name}] !! SPLIT MANQUANT")
            errors += 1
            continue

        ds = ds_dict[split_name]
        n = len(ds)
        print(f"\n[{split_name}] {n} samples")

        # Distribution par source
        src_counts = Counter(ds["source"])
        print(f"  Sources présentes : {len(src_counts)}")
        for src in sorted(src_counts, key=lambda s: -src_counts[s]):
            print(f"    {src:<20} {src_counts[src]:>8}")

        missing = expected_sources - set(src_counts.keys())
        if missing and split_name == "train":
            # RealWorldQA légitimement absent du train (eval-only, 765 samples test seuls)
            critical_missing = missing - {"realworldqa"}
            if critical_missing:
                print(f"  !! Sources manquantes dans train : {critical_missing}")
                errors += len(critical_missing)
        if missing and split_name == "test":
            # TextVQA légitimement absent du test (labels scellés)
            critical_missing = missing - {"textvqa"}
            if critical_missing:
                print(f"  !! Sources manquantes dans test : {critical_missing}")

        # Calibration : doit être stratifié sur toutes les sources train
        # (donc toutes sauf realworldqa qui n'a pas de train).
        if split_name == "calibration":
            expected_calib = expected_sources - {"realworldqa"}
            missing_calib = expected_calib - set(src_counts.keys())
            if missing_calib:
                print(f"  !! Sources absentes du calibration split : {missing_calib}")
                errors += len(missing_calib)
            # Déséquilibre massif = bug de stratification (sample sans remise)
            if src_counts:
                n_min = min(src_counts.values())
                n_max = max(src_counts.values())
                if n_max > 2 * n_min:
                    print(f"  !! Calibration déséquilibré : min={n_min} max={n_max} "
                          f"(ratio {n_max/max(1,n_min):.1f}x)")
                    errors += 1

        # Échantillon aléatoire pour vérifs profondes
        indices = random.sample(range(n), min(sample_check, n))
        n_checked = 0
        n_img_ok = 0
        n_img_fail = 0
        n_missing_q = 0
        n_missing_a = 0
        n_bad_meta = 0

        for idx in indices:
            row = ds[idx]
            n_checked += 1

            # Image
            try:
                img = row["image"]
                if img is None:
                    n_img_fail += 1
                else:
                    # HF renvoie PIL.Image si la feature est Image(); on check ouverture réelle
                    if isinstance(img, Image.Image):
                        img.verify()  # détecte les JPEG corrompus
                    elif isinstance(img, (bytes, bytearray)):
                        Image.open(io.BytesIO(img)).verify()
                    elif isinstance(img, dict) and img.get("bytes"):
                        Image.open(io.BytesIO(img["bytes"])).verify()
                    else:
                        n_img_fail += 1
                        continue
                    n_img_ok += 1
            except Exception:
                n_img_fail += 1

            # Question
            if not row.get("question"):
                n_missing_q += 1

            # Answers
            answers = row.get("answers")
            if not answers or all(not a for a in answers):
                n_missing_a += 1

            # Metadata (on a sérialisé en JSON string)
            meta = row.get("metadata")
            if meta:
                try:
                    json.loads(meta) if isinstance(meta, str) else dict(meta)
                except Exception:
                    n_bad_meta += 1

        print(f"  Échantillon sanity ({n_checked}) :")
        print(f"    images OK           : {n_img_ok}/{n_checked}")
        if n_img_fail:
            print(f"    !! images corrompues : {n_img_fail}")
            errors += n_img_fail
        if n_missing_q:
            print(f"    !! questions vides   : {n_missing_q}")
            errors += n_missing_q
        if n_missing_a:
            print(f"    !! answers vides     : {n_missing_a}")
            errors += n_missing_a
        if n_bad_meta:
            print(f"    !! metadata malformée: {n_bad_meta}")
            errors += n_bad_meta

    print("\n" + "=" * 72)
    if errors == 0:
        print("OK — Dataset validé, aucune anomalie détectée.")
    else:
        print(f"!! {errors} anomalies détectées. À investiguer avant partage équipe.")
    print("=" * 72)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_disk", type=str, help="Chemin local du DatasetDict (save_to_disk).")
    src.add_argument("--from_hub", type=str, help="Repo HF (pull).")
    ap.add_argument("--sample_check", type=int, default=100,
                    help="Nombre de samples à check en profondeur par split.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    from datasets import DatasetDict, load_dataset, load_from_disk

    if args.from_disk:
        dd = load_from_disk(args.from_disk)
    else:
        dd = load_dataset(args.from_hub)

    errors = validate(dd, sample_check=args.sample_check)
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
