#!/usr/bin/env python
"""
Orchestrateur de benchmark batch — lit configs/models.yaml et lance
eval_loop.py pour chaque modèle, soit individuellement soit en chaîne.

Usage (depuis la racine du projet) :

    # Lister les modèles disponibles dans le YAML
    python scripts/bench_all.py --list

    # Lancer un seul modèle (recommandé pour itérer)
    python scripts/bench_all.py --label gemma4-e4b-bf16

    # Lancer tous les modèles d'affilée (laisser tourner la nuit)
    python scripts/bench_all.py --all

    # Lancer un sous-ensemble par labels
    python scripts/bench_all.py --label gemma4-e4b-bf16 --label gemma4-e4b-bnb4

    # Override du subsample (full split avant soumission)
    python scripts/bench_all.py --all --subsample_pct 100

    # Continuer après un échec sans tout reprendre depuis le début
    python scripts/bench_all.py --all --skip_existing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "models.yaml"
EVAL_LOOP = ROOT / "scripts" / "eval_loop.py"
REPORTS_DIR = ROOT / "reports"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config introuvable : {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_model_table(models: list[dict]) -> None:
    print(f"{'LABEL':<28} {'BACKEND':<11} {'EXTRAS':<14} MODEL")
    print("-" * 100)
    for m in models:
        backend = m.get("backend", "transformers")
        extras = m.get("extras", "-")
        if backend == "llama_cpp":
            model_str = f"{m['gguf_repo']}/{m['gguf_filename']}"
        else:
            model_str = m["model_path"]
        print(f"{m['label']:<28} {backend:<11} {extras:<14} {model_str}")


def _build_eval_cmd(model: dict, defaults: dict, args) -> list[str]:
    label = model["label"]
    out_path = REPORTS_DIR / f"{label}.json"
    backend = model.get("backend", "transformers")

    cmd = [
        sys.executable, str(EVAL_LOOP),
        "--mode", "real",
        "--backend", backend,
        "--dataset_hub", args.dataset_hub or defaults["dataset_hub"],
        "--split", args.split or defaults["split"],
        "--subsample_pct", str(args.subsample_pct or defaults["subsample_pct"]),
        "--country_iso", args.country_iso or defaults["country_iso"],
        "--device", defaults.get("device", "cuda"),
        "--max_new_tokens", str(defaults.get("max_new_tokens", 32)),
        "--run_label", label,
        "--notes", model.get("notes", ""),
        "--out", str(out_path),
    ]

    if backend == "llama_cpp":
        cmd += [
            "--gguf_repo", model["gguf_repo"],
            "--gguf_filename", model["gguf_filename"],
            "--model_family", model.get("model_family", "gemma3"),
            "--n_gpu_layers", str(model.get("n_gpu_layers", -1)),
            "--n_ctx", str(model.get("n_ctx", 2048)),
        ]
        if model.get("mmproj_repo") and model.get("mmproj_filename"):
            cmd += [
                "--mmproj_repo", model["mmproj_repo"],
                "--mmproj_filename", model["mmproj_filename"],
            ]
    else:
        cmd += ["--model_path", model["model_path"]]

    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="Lister les modèles disponibles puis quitter.")
    ap.add_argument("--label", action="append", default=[],
                    help="Lancer uniquement le(s) modèle(s) avec ce label "
                         "(répétable). Mutex avec --all.")
    ap.add_argument("--all", action="store_true",
                    help="Lancer tous les modèles du YAML.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip les modèles dont reports/<label>.json existe déjà.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Affiche les commandes sans les exécuter.")

    # Overrides des defaults YAML
    ap.add_argument("--dataset_hub", default=None)
    ap.add_argument("--split", default=None, choices=["validation", "test", None])
    ap.add_argument("--subsample_pct", type=float, default=None)
    ap.add_argument("--country_iso", default=None)

    args = ap.parse_args()

    cfg = load_config()
    models = cfg["models"]
    defaults = cfg["defaults"]

    if args.list:
        _print_model_table(models)
        return 0

    if not args.all and not args.label:
        ap.error("--all ou --label requis (ou --list pour voir les options).")
    if args.all and args.label:
        ap.error("--all et --label sont mutuellement exclusifs.")

    if args.all:
        selected = models
    else:
        by_label = {m["label"]: m for m in models}
        unknown = [lab for lab in args.label if lab not in by_label]
        if unknown:
            raise SystemExit(
                f"Label(s) inconnu(s) : {unknown}. "
                f"Utiliser --list pour voir les options disponibles."
            )
        selected = [by_label[lab] for lab in args.label]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 72}")
    print(f"bench_all — {len(selected)} modèle(s) à évaluer")
    print(f"{'═' * 72}\n")

    failures: list[tuple[str, int]] = []
    for i, model in enumerate(selected, 1):
        label = model["label"]
        out_path = REPORTS_DIR / f"{label}.json"

        if args.skip_existing and out_path.exists():
            print(f"[{i}/{len(selected)}] SKIP {label} — {out_path} existe déjà")
            continue

        if model.get("backend") == "llama_cpp":
            model_str = f"{model['gguf_repo']}/{model['gguf_filename']}"
        else:
            model_str = model["model_path"]
        print(f"\n[{i}/{len(selected)}] ▶ {label}  ({model_str})")
        cmd = _build_eval_cmd(model, defaults, args)

        if args.dry_run:
            print("   DRY RUN :", " ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
        except subprocess.CalledProcessError as e:
            print(f"   ✗ ÉCHEC {label} (exit {e.returncode})")
            failures.append((label, e.returncode))
            # On continue avec les autres modèles plutôt que tout arrêter.
            continue
        except KeyboardInterrupt:
            print("\nInterruption utilisateur — arrêt.")
            return 130

    print(f"\n{'═' * 72}")
    if failures:
        print(f"Terminé avec {len(failures)} échec(s) :")
        for label, code in failures:
            print(f"   - {label} (exit {code})")
    else:
        print("Toutes les runs sont passées sans erreur.")
    print(f"Registry : results/registry.csv")
    print(f"{'═' * 72}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
