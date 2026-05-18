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

    # Lancer un groupe YAML (ex: smoke vLLM)
    python scripts/bench_all.py --group vllm_smoke

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


def _print_groups(groups: dict[str, list[str]]) -> None:
    if not groups:
        return
    print("\nGROUPS")
    print("-" * 100)
    for name, labels in groups.items():
        print(f"{name:<32} {', '.join(labels)}")


def _build_eval_cmd(model: dict, defaults: dict, args) -> list[str]:
    label = model["label"]
    out_path = REPORTS_DIR / f"{label}.json"
    backend = model.get("backend", "transformers")

    cmd = [
        sys.executable, str(EVAL_LOOP),
        "--mode", "real",
        "--backend", backend,
        "--split", args.split or defaults["split"],
        "--subsample_pct", str(args.subsample_pct or defaults["subsample_pct"]),
        "--country_iso", args.country_iso or defaults["country_iso"],
        "--device", defaults.get("device", "cuda"),
        "--max_new_tokens", str(model.get("max_new_tokens", defaults.get("max_new_tokens", 32))),
        "--temperature", str(model.get("temperature", defaults.get("temperature", 1.0))),
        "--top_p", str(model.get("top_p", defaults.get("top_p", 0.95))),
        "--top_k", str(model.get("top_k", defaults.get("top_k", 64))),
        "--run_label", label,
        "--notes", model.get("notes", ""),
        "--out", str(out_path),
    ]

    # Dataset : local (chemin disque) prioritaire sur hub HF si fourni
    if args.dataset_local:
        cmd += ["--dataset_local", args.dataset_local]
    else:
        cmd += ["--dataset_hub", args.dataset_hub or defaults["dataset_hub"]]

    # Cap dur du nombre de samples par source (smoke test / itération rapide)
    if args.limit_per_source is not None:
        cmd += ["--limit_per_source", str(args.limit_per_source)]

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
    elif backend == "vllm":
        cmd += ["--model_path", model["model_path"]]
        vllm_map = {
            "vllm_tensor_parallel_size": "--vllm_tensor_parallel_size",
            "vllm_max_model_len": "--vllm_max_model_len",
            "vllm_gpu_memory_utilization": "--vllm_gpu_memory_utilization",
            "vllm_max_num_seqs": "--vllm_max_num_seqs",
            "vllm_dtype": "--vllm_dtype",
            "vllm_quantization": "--vllm_quantization",
            "vllm_kv_cache_dtype": "--vllm_kv_cache_dtype",
            "vllm_max_soft_tokens": "--vllm_max_soft_tokens",
            "vllm_seed": "--vllm_seed",
        }
        for key, flag in vllm_map.items():
            value = model.get(key, defaults.get(key))
            if value is not None:
                cmd += [flag, str(value)]
        if model.get("vllm_enforce_eager", defaults.get("vllm_enforce_eager", False)):
            cmd += ["--vllm_enforce_eager"]
        if model.get("revision"):
            cmd += ["--revision", model["revision"]]
    else:
        cmd += ["--model_path", model["model_path"]]

    return cmd


def _cleanup_model_cache(model: dict) -> None:
    """Supprime le cache HF du modèle (libère 5-15 GB) pour permettre au
    suivant de tenir dans la quota disque. Cherche dans HF_HOME (override
    env) puis dans le défaut HuggingFace.
    """
    import os
    import shutil

    repos: list[str] = []
    if model.get("backend") == "llama_cpp":
        repos.append(model["gguf_repo"])
        if model.get("mmproj_repo") and model["mmproj_repo"] != model["gguf_repo"]:
            repos.append(model["mmproj_repo"])
    else:
        repos.append(model["model_path"])

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"

    for repo in repos:
        slug = "models--" + repo.replace("/", "--")
        path = hub_dir / slug
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"   [cleanup] removed {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="Lister les modèles disponibles puis quitter.")
    ap.add_argument("--label", action="append", default=[],
                    help="Lancer uniquement le(s) modèle(s) avec ce label "
                         "(répétable). Mutex avec --all.")
    ap.add_argument("--group", action="append", default=[],
                    help="Lancer un groupe défini dans configs/models.yaml "
                         "(répétable). Mutex avec --all.")
    ap.add_argument("--all", action="store_true",
                    help="Lancer tous les modèles du YAML.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip les modèles dont reports/<label>.json existe déjà.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Affiche les commandes sans les exécuter.")

    # Overrides des defaults YAML
    ap.add_argument("--dataset_hub", default=None,
                    help="Override le dataset HF Hub du YAML.")
    ap.add_argument("--dataset_local", default=None,
                    help="Charger un DatasetDict local (save_to_disk). "
                         "Prioritaire sur --dataset_hub si fourni.")
    ap.add_argument("--split", default=None, choices=["validation", "test", None])
    ap.add_argument("--subsample_pct", type=float, default=None)
    ap.add_argument("--limit_per_source", type=int, default=None,
                    help="Cap dur du nombre de samples par source (post-subsample). "
                         "Utile pour smoke test rapide (~10 min/modèle avec 20).")
    ap.add_argument("--country_iso", default=None)
    ap.add_argument("--cleanup_cache_after", action="store_true",
                    help="Supprime le cache HF du modèle après chaque run "
                         "(libère 5-15 GB pour permettre au suivant de tenir).")

    args = ap.parse_args()

    cfg = load_config()
    models = cfg["models"]
    defaults = cfg["defaults"]
    groups = cfg.get("groups", {})

    if args.list:
        _print_model_table(models)
        _print_groups(groups)
        return 0

    if not args.all and not args.label and not args.group:
        ap.error("--all, --label ou --group requis (ou --list pour voir les options).")
    if args.all and (args.label or args.group):
        ap.error("--all est mutuellement exclusif avec --label / --group.")

    if args.all:
        selected = [m for m in models if m.get("run_by_default", True)]
    elif args.label:
        by_label = {m["label"]: m for m in models}
        unknown = [lab for lab in args.label if lab not in by_label]
        if unknown:
            raise SystemExit(
                f"Label(s) inconnu(s) : {unknown}. "
                f"Utiliser --list pour voir les options disponibles."
            )
        selected = [by_label[lab] for lab in args.label]
    else:
        by_label = {m["label"]: m for m in models}
        unknown_groups = [name for name in args.group if name not in groups]
        if unknown_groups:
            raise SystemExit(
                f"Groupe(s) inconnu(s) : {unknown_groups}. "
                f"Utiliser --list pour voir les groupes disponibles."
            )
        labels: list[str] = []
        for name in args.group:
            labels.extend(groups[name])
        unknown_labels = [lab for lab in labels if lab not in by_label]
        if unknown_labels:
            raise SystemExit(
                f"Label(s) de groupe inconnu(s) : {unknown_labels}. "
                "Corriger configs/models.yaml."
            )
        selected = [by_label[lab] for lab in dict.fromkeys(labels)]

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
        except KeyboardInterrupt:
            print("\nInterruption utilisateur — arrêt.")
            return 130

        # Cleanup cache HF du modèle qu'on vient d'évaluer (réussi ou pas)
        # pour libérer la place avant le suivant.
        if args.cleanup_cache_after:
            _cleanup_model_cache(model)

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
