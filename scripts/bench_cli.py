#!/usr/bin/env python
"""CLI ergonomique pour lancer les benchmarks Runpod.

Cette CLI ne remplace pas `bench_all.py` : elle construit des commandes
lisibles et les délègue au runner existant.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    import questionary
except ImportError:  # pragma: no cover - message runtime plus utile
    questionary = None


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "models.yaml"
BENCH_ALL = ROOT / "scripts" / "bench_all.py"
EVAL_LOOP = ROOT / "scripts" / "eval_loop.py"
DEFAULT_DATASET_HUB = "ECE-ILAB/resilient-ai-unified"
RUNPOD_DATASET_LOCAL = Path("/workspace/code/data/unified")
DEFAULT_COUNTRY = "CZE"
DEFAULT_SPLIT = "test"

app = typer.Typer(
    add_completion=False,
    help="CLI Runpod pour lancer les benchmarks Resilient AI.",
)
console = Console(width=140)


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise typer.BadParameter(f"Config introuvable: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _models_by_label(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["label"]: m for m in cfg.get("models", [])}


def _models_matching_label(cfg: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [m for m in cfg.get("models", []) if m["label"] == label]


def _duplicate_labels(cfg: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model in cfg.get("models", []):
        label = model["label"]
        counts[label] = counts.get(label, 0) + 1
    return {label: count for label, count in counts.items() if count > 1}


def _groups(cfg: dict[str, Any]) -> dict[str, list[str]]:
    return cfg.get("groups", {}) or {}


def _defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("defaults", {}) or {}


def _normalize_hf_dataset(value: str) -> str:
    """Accepte un repo id HF ou une URL Hugging Face dataset."""
    raw = value.strip()
    parsed = urlparse(raw)
    if parsed.netloc in {"huggingface.co", "www.huggingface.co"}:
        parts = [p for p in parsed.path.split("/") if p]
        if parts[:1] == ["datasets"] and len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return raw


def _normalize_hf_repo(value: str) -> str:
    """Accepte un repo id HF, une URL HF modele/dataset, ou un chemin local."""
    raw = value.strip()
    parsed = urlparse(raw)
    if parsed.netloc in {"huggingface.co", "www.huggingface.co"}:
        parts = [p for p in parsed.path.split("/") if p]
        if parts[:1] == ["datasets"] and len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return raw


def _runpod_dataset_exists() -> bool:
    return RUNPOD_DATASET_LOCAL.exists()


def _dataset_args(
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    prefer_local: bool,
    smoke: bool,
    streaming_take: int | None,
    streaming_per_source: int | None,
) -> list[str]:
    if dataset_local:
        return ["--dataset_local", str(dataset_local)]
    if dataset_hub:
        args = ["--dataset_hub", _normalize_hf_dataset(dataset_hub)]
    elif prefer_local and _runpod_dataset_exists():
        return ["--dataset_local", str(RUNPOD_DATASET_LOCAL)]
    else:
        args = ["--dataset_hub", DEFAULT_DATASET_HUB]

    if streaming_per_source is not None:
        args += ["--dataset_streaming_per_source", str(streaming_per_source)]
    elif smoke and streaming_take is not None:
        args += ["--dataset_streaming_take", str(streaming_take)]
    return args


def _direct_dataset_args(
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    prefer_local: bool,
    streaming_take: int | None,
    streaming_per_source: int | None,
    fallback_streaming_per_source: int | None,
) -> list[str]:
    if dataset_local:
        return ["--dataset_local", str(dataset_local)]
    if dataset_hub:
        args = ["--dataset_hub", _normalize_hf_dataset(dataset_hub)]
    elif prefer_local and _runpod_dataset_exists():
        return ["--dataset_local", str(RUNPOD_DATASET_LOCAL)]
    else:
        args = ["--dataset_hub", DEFAULT_DATASET_HUB]

    if streaming_per_source is not None:
        args += ["--dataset_streaming_per_source", str(streaming_per_source)]
    elif streaming_take is not None:
        args += ["--dataset_streaming_take", str(streaming_take)]
    elif fallback_streaming_per_source is not None and not _runpod_dataset_exists():
        args += ["--dataset_streaming_per_source", str(fallback_streaming_per_source)]
    return args


def _base_cmd(
    *,
    country: str,
    subsample_pct: float,
    split: str,
    dataset_hub: str | None,
    dataset_local: Path | None,
    prefer_local: bool,
    smoke: bool,
    streaming_take: int | None,
    streaming_per_source: int | None,
) -> list[str]:
    return [
        sys.executable,
        str(BENCH_ALL),
        "--country_iso",
        country,
        "--split",
        split,
        "--subsample_pct",
        str(subsample_pct),
        *_dataset_args(
            dataset_hub=dataset_hub,
            dataset_local=dataset_local,
            prefer_local=prefer_local,
            smoke=smoke,
            streaming_take=streaming_take,
            streaming_per_source=streaming_per_source,
        ),
    ]


def _append_runner_flags(
    cmd: list[str],
    *,
    skip_existing: bool,
    cleanup_cache: bool,
) -> list[str]:
    if skip_existing:
        cmd.append("--skip_existing")
    if cleanup_cache:
        cmd.append("--cleanup_cache_after")
    return cmd


def _print_command(cmd: list[str]) -> None:
    console.print("[bold]Commande:[/bold]")
    console.print(shlex.join(cmd))


def _run(cmd: list[str], dry_run: bool, print_cmd: bool = True) -> None:
    if print_cmd:
        _print_command(cmd)
    if dry_run:
        return
    raise typer.Exit(subprocess.run(cmd, cwd=ROOT).returncode)


def _parse_limit(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise typer.BadParameter("Limit doit etre un entier ou vide.") from exc
    if limit <= 0:
        raise typer.BadParameter("Limit doit etre positif.")
    return limit


def _parse_sources(value: str | None) -> list[str] | None:
    if not value:
        return None
    sources = [part.strip() for part in value.split(",") if part.strip()]
    return sources or None


def _append_sources(cmd: list[str], sources: str | None) -> list[str]:
    parsed = _parse_sources(sources)
    if parsed:
        cmd += ["--sources", *parsed]
    return cmd


def _smoke_cmd(
    backend: str,
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    limit: int,
    country: str,
    split: str,
    streaming_take: int,
    streaming_per_source: int | None,
    skip_existing: bool,
    cleanup_cache: bool,
) -> list[str]:
    cfg = _load_config()
    cmd = _base_cmd(
        country=country,
        subsample_pct=100,
        split=split,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        prefer_local=False,
        smoke=True,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
    )
    if backend == "vllm":
        if "vllm_smoke" not in _groups(cfg):
            raise typer.BadParameter("Groupe vllm_smoke absent de configs/models.yaml.")
        cmd += ["--group", "vllm_smoke"]
    elif backend == "llama_cpp":
        cmd += ["--label", "gemma4-e4b-gguf-q4km"]
    else:
        raise typer.BadParameter("Backend supporte: vllm, llama_cpp.")
    cmd += ["--limit_per_source", str(limit)]
    return _append_runner_flags(cmd, skip_existing=skip_existing, cleanup_cache=cleanup_cache)


def _label_cmd(
    label: str,
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    limit: int | None,
    country: str,
    split: str,
    subsample_pct: float,
    streaming_take: int | None,
    streaming_per_source: int | None,
    skip_existing: bool,
    cleanup_cache: bool,
) -> list[str]:
    cfg = _load_config()
    if label not in _models_by_label(cfg):
        raise typer.BadParameter(f"Label inconnu: {label}")
    cmd = _base_cmd(
        country=country,
        subsample_pct=subsample_pct,
        split=split,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        prefer_local=True,
        smoke=False,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
    )
    cmd += ["--label", label]
    if limit is not None:
        cmd += ["--limit_per_source", str(limit)]
    return _append_runner_flags(cmd, skip_existing=skip_existing, cleanup_cache=cleanup_cache)


def _group_cmd(
    group_name: str,
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    limit: int | None,
    country: str,
    split: str,
    subsample_pct: float,
    streaming_take: int | None,
    streaming_per_source: int | None,
    skip_existing: bool,
    cleanup_cache: bool,
) -> list[str]:
    cfg = _load_config()
    if group_name not in _groups(cfg):
        raise typer.BadParameter(f"Groupe inconnu: {group_name}")
    cmd = _base_cmd(
        country=country,
        subsample_pct=subsample_pct,
        split=split,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        prefer_local=True,
        smoke=False,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
    )
    cmd += ["--group", group_name]
    if limit is not None:
        cmd += ["--limit_per_source", str(limit)]
    return _append_runner_flags(cmd, skip_existing=skip_existing, cleanup_cache=cleanup_cache)


def _labels_cmd(
    labels: list[str],
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    limit: int | None,
    country: str,
    split: str,
    subsample_pct: float,
    streaming_take: int | None,
    streaming_per_source: int | None,
    skip_existing: bool,
    cleanup_cache: bool,
) -> list[str]:
    cfg = _load_config()
    by_label = _models_by_label(cfg)
    unknown = [label for label in labels if label not in by_label]
    if unknown:
        raise typer.BadParameter(f"Labels inconnus: {', '.join(unknown)}")
    if not labels:
        raise typer.BadParameter("Au moins un label est requis.")
    cmd = _base_cmd(
        country=country,
        subsample_pct=subsample_pct,
        split=split,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        prefer_local=True,
        smoke=False,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
    )
    for label in labels:
        cmd += ["--label", label]
    if limit is not None:
        cmd += ["--limit_per_source", str(limit)]
    return _append_runner_flags(cmd, skip_existing=skip_existing, cleanup_cache=cleanup_cache)


def _final_cmd(
    target: str,
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    country: str,
    split: str,
    streaming_take: int | None,
    streaming_per_source: int | None,
    skip_existing: bool,
    cleanup_cache: bool,
) -> list[str]:
    cfg = _load_config()
    cmd = _base_cmd(
        country=country,
        subsample_pct=100,
        split=split,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        prefer_local=True,
        smoke=False,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
    )
    if target in _groups(cfg):
        cmd += ["--group", target]
    elif target in _models_by_label(cfg):
        cmd += ["--label", target]
    else:
        raise typer.BadParameter(f"Label ou groupe inconnu: {target}")
    return _append_runner_flags(cmd, skip_existing=skip_existing, cleanup_cache=cleanup_cache)


def _all_cmd(
    *,
    dataset_hub: str | None,
    dataset_local: Path | None,
    limit: int | None,
    country: str,
    split: str,
    subsample_pct: float,
    streaming_take: int | None,
    streaming_per_source: int | None,
    skip_existing: bool,
    cleanup_cache: bool,
) -> list[str]:
    cmd = _base_cmd(
        country=country,
        subsample_pct=subsample_pct,
        split=split,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        prefer_local=True,
        smoke=False,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
    )
    cmd.append("--all")
    if limit is not None:
        cmd += ["--limit_per_source", str(limit)]
    return _append_runner_flags(cmd, skip_existing=skip_existing, cleanup_cache=cleanup_cache)


def _safe_label(prefix: str, value: str) -> str:
    stem = _normalize_hf_repo(value).rstrip("/").split("/")[-1] or value
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-").lower()
    return f"{prefix}-{slug or 'model'}"


def _direct_eval_base_cmd(
    *,
    backend: str,
    label: str,
    dataset_hub: str | None,
    dataset_local: Path | None,
    limit: int | None,
    country: str,
    split: str,
    subsample_pct: float,
    streaming_take: int | None,
    streaming_per_source: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    no_energy_tracking: bool,
    notes: str,
) -> list[str]:
    out_path = ROOT / "reports" / f"{label}.json"
    cmd = [
        sys.executable,
        str(EVAL_LOOP),
        "--mode",
        "real",
        "--backend",
        backend,
        "--split",
        split,
        "--subsample_pct",
        str(subsample_pct),
        "--country_iso",
        country,
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
        "--run_label",
        label,
        "--notes",
        notes,
        "--out",
        str(out_path),
        *_direct_dataset_args(
            dataset_hub=dataset_hub,
            dataset_local=dataset_local,
            prefer_local=True,
            streaming_take=streaming_take,
            streaming_per_source=streaming_per_source,
            fallback_streaming_per_source=limit,
        ),
    ]
    if limit is not None:
        cmd += ["--limit_per_source", str(limit)]
    if no_energy_tracking:
        cmd.append("--no_energy_tracking")
    return cmd


def _adhoc_model(label: str, backend: str, model: str, runtime: str, notes: str) -> dict[str, Any]:
    return {
        "label": label,
        "backend": backend,
        "model_path": model,
        "notes": notes,
        "_runtime": runtime,
    }


def _model_ref(model: dict[str, Any]) -> str:
    if "_model_ref" in model:
        return str(model["_model_ref"])
    backend = model.get("backend", "transformers")
    if backend == "llama_cpp":
        return f"{model.get('gguf_repo', '')}/{model.get('gguf_filename', '')}"
    return model.get("model_path", "")


def _model_runtime(model: dict[str, Any], cfg: dict[str, Any]) -> str:
    if "_runtime" in model:
        return str(model["_runtime"])
    backend = model.get("backend", "transformers")
    defaults = _defaults(cfg)
    if backend == "vllm":
        bits = [
            f"soft={model.get('vllm_max_soft_tokens', defaults.get('vllm_max_soft_tokens', '-'))}",
            f"ctx={model.get('vllm_max_model_len', defaults.get('vllm_max_model_len', '-'))}",
            f"quant={model.get('vllm_quantization', '-')}",
            f"kv={model.get('vllm_kv_cache_dtype', '-')}",
            f"tok={model.get('max_new_tokens', defaults.get('max_new_tokens', '-'))}",
        ]
        return " ".join(bits)
    if backend == "llama_cpp":
        return (
            f"family={model.get('model_family', 'gemma3')} "
            f"ctx={model.get('n_ctx', 2048)} "
            f"gpu_layers={model.get('n_gpu_layers', -1)}"
        )
    return f"tok={model.get('max_new_tokens', defaults.get('max_new_tokens', 32))}"


def _selected_models_for_target(
    cfg: dict[str, Any],
    *,
    label: str | None = None,
    labels: list[str] | None = None,
    group: str | None = None,
    all_default: bool = False,
) -> list[dict[str, Any]]:
    by_label = _models_by_label(cfg)
    if all_default:
        return [m for m in cfg.get("models", []) if m.get("run_by_default", True)]
    if group:
        names = _groups(cfg).get(group)
        if names is None:
            raise typer.BadParameter(f"Groupe inconnu: {group}")
        return [by_label[name] for name in dict.fromkeys(names)]
    if labels:
        return [by_label[name] for name in labels]
    if label:
        return [by_label[label]]
    return []


def _print_run_plan(
    *,
    title: str,
    cmd: list[str],
    models: list[dict[str, Any]],
    cfg: dict[str, Any],
    dry_run: bool,
) -> None:
    console.print(Panel(title, title="Run plan"))
    backend_counts: dict[str, int] = {}
    for model in models:
        backend = model.get("backend", "transformers")
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
    if backend_counts:
        console.print(
            "[bold]Selection:[/bold] "
            + ", ".join(f"{backend}={count}" for backend, count in sorted(backend_counts.items()))
            + f"  total={len(models)}"
        )
    table = Table(title="Selected models", expand=True)
    table.add_column("label", overflow="ellipsis", no_wrap=True, ratio=2)
    table.add_column("backend", overflow="ellipsis", no_wrap=True, width=12)
    table.add_column("model", overflow="ellipsis", no_wrap=True, ratio=2)
    table.add_column("runtime", overflow="ellipsis", no_wrap=True, ratio=2)
    table.add_column("notes", overflow="ellipsis", no_wrap=True, ratio=3)
    max_rows = 18
    for model in models[:max_rows]:
        table.add_row(
            model["label"],
            model.get("backend", "transformers"),
            _short(_model_ref(model), 42),
            _short(_model_runtime(model, cfg), 42),
            _short(model.get("notes", ""), 60),
        )
    console.print(table)
    if len(models) > max_rows:
        console.print(f"[cyan]... {len(models) - max_rows} autres modeles masques dans ce plan.[/cyan]")
    dupes = _duplicate_labels(cfg)
    selected_labels = {model["label"] for model in models}
    relevant_dupes = {label: count for label, count in dupes.items() if label in selected_labels}
    if relevant_dupes:
        console.print(
            "[yellow]Config warning:[/yellow] labels selectionnes dupliques dans models.yaml: "
            + ", ".join(sorted(relevant_dupes))
        )
        console.print("[yellow]bench_all garde la derniere entree pour un label duplique.[/yellow]")
    _print_command(cmd)
    if dry_run:
        console.print("[cyan]Dry-run: rien n'est lance.[/cyan]")


def _infer_backend(row: dict[str, str]) -> str:
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    if cfg.get("backend"):
        return str(cfg["backend"])
    label = row.get("run_label", "").lower()
    if "vllm" in label:
        return "vllm"
    if "gguf" in label:
        return "llama_cpp"
    return "transformers"


def _float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def _require_questionary() -> None:
    if questionary is None:
        raise typer.BadParameter(
            "questionary n'est pas installe. Lance `pip install questionary` "
            "ou `pip install -r requirements.txt`."
        )


def _choice(title: str, value: str):
    return questionary.Choice(title=title, value=value)


def _ok(value: bool) -> str:
    return "ok" if value else "missing"


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_dataset_specs():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import ALL_DATASETS
    return ALL_DATASETS


def _short(value: str, limit: int = 26) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _num(row: dict[str, str], key: str, digits: int = 4) -> str:
    raw = row.get(key, "")
    if raw == "":
        return ""
    try:
        return f"{float(raw):.{digits}g}"
    except ValueError:
        return raw


@app.command("list")
def list_models(
    backend: str | None = typer.Option(None, help="Filtre backend."),
    query: str | None = typer.Option(None, help="Filtre texte sur label/model."),
    default_only: bool = typer.Option(False, help="Afficher seulement run_by_default=true."),
) -> None:
    """Liste les modèles et groupes disponibles."""
    cfg = _load_config()
    table = Table(title="Models")
    table.add_column("Label", style="bold")
    table.add_column("Backend")
    table.add_column("Default")
    table.add_column("Model")
    for model in cfg.get("models", []):
        model_backend = model.get("backend", "transformers")
        model_ref = (
            f"{model.get('gguf_repo', '')}/{model.get('gguf_filename', '')}"
            if model_backend == "llama_cpp"
            else model.get("model_path", "")
        )
        haystack = f"{model['label']} {model_ref}".lower()
        if backend and model_backend != backend:
            continue
        if query and query.lower() not in haystack:
            continue
        if default_only and not model.get("run_by_default", True):
            continue
        table.add_row(
            model["label"],
            model_backend,
            "yes" if model.get("run_by_default", True) else "no",
            model_ref,
        )
    console.print(table)

    group_table = Table(title="Groups")
    group_table.add_column("Group", style="bold")
    group_table.add_column("Labels")
    by_label = _models_by_label(cfg)
    for name, labels in _groups(cfg).items():
        visible = []
        for label in labels:
            model = by_label.get(label)
            model_backend = model.get("backend", "transformers") if model else "missing"
            haystack = f"{name} {label}".lower()
            if backend and model_backend != backend:
                continue
            if query and query.lower() not in haystack:
                continue
            visible.append(label)
        if backend or query:
            if not visible:
                continue
            group_table.add_row(name, ", ".join(visible))
        else:
            group_table.add_row(name, ", ".join(labels))
    console.print(group_table)


@app.command("model")
def model_detail(
    label: str = typer.Argument(..., help="Label modele dans configs/models.yaml."),
) -> None:
    """Affiche la config détaillée d'un modèle."""
    cfg = _load_config()
    matches = _models_matching_label(cfg, label)
    if not matches:
        raise typer.BadParameter(f"Label inconnu: {label}")
    if len(matches) > 1:
        console.print(
            f"[yellow]{len(matches)} entrees portent ce label. bench_all utilisera la derniere.[/yellow]"
        )
    for idx, model in enumerate(matches, 1):
        table = Table(title=f"{label} #{idx}")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for key, value in model.items():
            table.add_row(str(key), json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
        table.add_row("resolved_backend", model.get("backend", "transformers"))
        table.add_row("resolved_model", _model_ref(model))
        table.add_row("runtime", _model_runtime(model, cfg))
        console.print(table)


@app.command("sources")
def sources() -> None:
    """Liste les sources du dataset unifié avec tâches et métriques."""
    specs = _load_dataset_specs()
    table = Table(title="Benchmark sources")
    for column in ("source", "tier", "task", "metric", "hf_id", "splits"):
        table.add_column(column, overflow="ellipsis", no_wrap=(column != "splits"))
    for name, spec in sorted(specs.items(), key=lambda item: (item[1].tier, item[0])):
        splits = ", ".join(f"{k}={v or '-'}" for k, v in spec.splits.items())
        table.add_row(
            name,
            str(spec.tier),
            spec.task,
            spec.metric,
            _short(spec.hf_id, 36),
            splits,
        )
    console.print(table)


@app.command("explain")
def explain() -> None:
    """Explique le flow du benchmark actuel."""
    cfg = _load_config()
    specs = _load_dataset_specs()
    defaults = _defaults(cfg)
    console.print(
        Panel(
            "bench_all lit configs/models.yaml, selectionne labels/groupes/all, "
            "puis lance eval_loop une fois par modele. eval_loop charge le dataset, "
            "construit le backend, evalue par source, track CodeCarbon, puis append "
            "results/registry.csv.",
            title="Flow",
        )
    )
    defaults_table = Table(title="Defaults YAML")
    defaults_table.add_column("Key", style="bold")
    defaults_table.add_column("Value")
    for key, value in defaults.items():
        defaults_table.add_row(str(key), str(value))
    console.print(defaults_table)

    metrics = Table(title="Metrics by source")
    metrics.add_column("source", style="bold")
    metrics.add_column("task")
    metrics.add_column("metric")
    metrics.add_column("why")
    why = {
        "anls": "OCR/doc tolerant",
        "vqa_accuracy": "VQA consensus/exact",
        "relaxed_accuracy": "numeric +/-5%",
        "multiple_choice": "letter match",
        "realworld_accuracy": "MC or normalized text",
        "cider": "caption corpus score",
    }
    for name, spec in sorted(specs.items()):
        metrics.add_row(name, spec.task, spec.metric, why.get(spec.metric, ""))
    console.print(metrics)

    backends = Table(title="Backends")
    backends.add_column("backend", style="bold")
    backends.add_column("format")
    backends.add_column("path in code")
    backends.add_row("transformers", "HF safetensors / AWQ", "eval_loop.py + make_gemma_generate_fn")
    backends.add_row("vllm", "HF safetensors via offline LLM", "src/models/vllm_backend.py")
    backends.add_row("llama_cpp", "GGUF + mmproj", "src/models/llama_cpp_backend.py")
    console.print(backends)


@app.command()
def smoke(
    backend: str = typer.Argument(..., help="Backend: vllm ou llama_cpp."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int = typer.Option(2, help="Nombre de samples par source."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    streaming_take: int = typer.Option(64, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    skip_existing: bool = typer.Option(False, help="Skip reports deja presents."),
    cleanup_cache: bool = typer.Option(False, help="Nettoie le cache HF apres run."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Lance un smoke test rapide."""
    cfg = _load_config()
    cmd = _smoke_cmd(
        backend,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        skip_existing=skip_existing,
        cleanup_cache=cleanup_cache,
    )
    models = _selected_models_for_target(
        cfg,
        group="vllm_smoke" if backend == "vllm" else None,
        label="gemma4-e4b-gguf-q4km" if backend == "llama_cpp" else None,
    )
    _print_run_plan(title=f"smoke {backend}", cmd=cmd, models=models, cfg=cfg, dry_run=dry_run)
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("run")
def run_label(
    label: str = typer.Argument(..., help="Label modele dans configs/models.yaml."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int | None = typer.Option(20, help="Samples par source. Vide pour aucun cap."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    subsample_pct: float = typer.Option(100, help="Pourcentage du split."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    skip_existing: bool = typer.Option(False, help="Skip reports deja presents."),
    cleanup_cache: bool = typer.Option(False, help="Nettoie le cache HF apres run."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Lance un modèle par label."""
    cfg = _load_config()
    cmd = _label_cmd(
        label,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        subsample_pct=subsample_pct,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        skip_existing=skip_existing,
        cleanup_cache=cleanup_cache,
    )
    _print_run_plan(
        title=f"run {label}",
        cmd=cmd,
        models=_selected_models_for_target(cfg, label=label),
        cfg=cfg,
        dry_run=dry_run,
    )
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("group")
def run_group(
    group_name: str = typer.Argument(..., help="Groupe dans configs/models.yaml."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int | None = typer.Option(20, help="Samples par source. Vide pour aucun cap."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    subsample_pct: float = typer.Option(100, help="Pourcentage du split."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    skip_existing: bool = typer.Option(False, help="Skip reports deja presents."),
    cleanup_cache: bool = typer.Option(False, help="Nettoie le cache HF apres run."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Lance un groupe de modèles."""
    cfg = _load_config()
    cmd = _group_cmd(
        group_name,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        subsample_pct=subsample_pct,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        skip_existing=skip_existing,
        cleanup_cache=cleanup_cache,
    )
    _print_run_plan(
        title=f"group {group_name}",
        cmd=cmd,
        models=_selected_models_for_target(cfg, group=group_name),
        cfg=cfg,
        dry_run=dry_run,
    )
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("batch")
def run_batch(
    labels: list[str] = typer.Argument(..., help="Labels modeles a lancer."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int | None = typer.Option(20, help="Samples par source. Vide pour aucun cap."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    subsample_pct: float = typer.Option(100, help="Pourcentage du split."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    skip_existing: bool = typer.Option(False, help="Skip reports deja presents."),
    cleanup_cache: bool = typer.Option(False, help="Nettoie le cache HF apres run."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Lance plusieurs labels sans créer de groupe YAML."""
    cfg = _load_config()
    cmd = _labels_cmd(
        labels,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        subsample_pct=subsample_pct,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        skip_existing=skip_existing,
        cleanup_cache=cleanup_cache,
    )
    _print_run_plan(
        title=f"batch {len(labels)} labels",
        cmd=cmd,
        models=_selected_models_for_target(cfg, labels=labels),
        cfg=cfg,
        dry_run=dry_run,
    )
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("all")
def run_all(
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int | None = typer.Option(20, help="Samples par source. Vide pour aucun cap."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    subsample_pct: float = typer.Option(100, help="Pourcentage du split."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    skip_existing: bool = typer.Option(True, help="Skip reports deja presents."),
    cleanup_cache: bool = typer.Option(True, help="Nettoie le cache HF apres run."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Lance tous les modèles run_by_default du YAML."""
    cfg = _load_config()
    cmd = _all_cmd(
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        subsample_pct=subsample_pct,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        skip_existing=skip_existing,
        cleanup_cache=cleanup_cache,
    )
    _print_run_plan(
        title="all run_by_default",
        cmd=cmd,
        models=_selected_models_for_target(cfg, all_default=True),
        cfg=cfg,
        dry_run=dry_run,
    )
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("try-vllm")
def try_vllm(
    model_path: str = typer.Argument(..., help="Repo HF ou chemin local safetensors."),
    label: str | None = typer.Option(None, help="Run label. Defaut derive du model_path."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int | None = typer.Option(20, help="Samples par source. Vide pour aucun cap."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    subsample_pct: float = typer.Option(100, help="Pourcentage du split."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    source: str | None = typer.Option(None, help="Sources separees par virgule, ex: docvqa,textvqa."),
    max_new_tokens: int = typer.Option(16, help="Tokens generes."),
    temperature: float = typer.Option(1.0, help="Sampling temperature."),
    top_p: float = typer.Option(0.95, help="top_p."),
    top_k: int = typer.Option(64, help="top_k."),
    tensor_parallel_size: int = typer.Option(1, help="vLLM tensor parallel size."),
    max_model_len: int = typer.Option(2048, help="vLLM max model len."),
    gpu_memory_utilization: float = typer.Option(0.90, help="vLLM GPU memory utilization."),
    max_num_seqs: int = typer.Option(1, help="vLLM max num seqs."),
    dtype: str = typer.Option("auto", help="vLLM dtype."),
    quantization: str | None = typer.Option(None, help="vLLM quantization, ex: fp8, awq."),
    kv_cache_dtype: str | None = typer.Option(None, help="vLLM KV cache dtype, ex: fp8."),
    max_soft_tokens: int | None = typer.Option(None, help="Gemma image max_soft_tokens."),
    enforce_eager: bool = typer.Option(False, help="Passe --vllm_enforce_eager."),
    revision: str | None = typer.Option(None, help="Revision HF."),
    seed: int = typer.Option(42, help="vLLM sampling seed."),
    disable_energy_tracking: bool = typer.Option(False, "--no-energy-tracking", help="Desactive CodeCarbon."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Teste un repo safetensors/HF via vLLM sans l'ajouter au YAML."""
    cfg = _load_config()
    model_path = _normalize_hf_repo(model_path)
    run_label = label or _safe_label("vllm", model_path)
    notes = "adhoc vLLM"
    cmd = _direct_eval_base_cmd(
        backend="vllm",
        label=run_label,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        subsample_pct=subsample_pct,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        no_energy_tracking=disable_energy_tracking,
        notes=notes,
    )
    cmd += [
        "--model_path",
        model_path,
        "--vllm_tensor_parallel_size",
        str(tensor_parallel_size),
        "--vllm_max_model_len",
        str(max_model_len),
        "--vllm_gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--vllm_max_num_seqs",
        str(max_num_seqs),
        "--vllm_dtype",
        dtype,
        "--vllm_seed",
        str(seed),
    ]
    if quantization:
        cmd += ["--vllm_quantization", quantization]
    if kv_cache_dtype:
        cmd += ["--vllm_kv_cache_dtype", kv_cache_dtype]
    if max_soft_tokens is not None:
        cmd += ["--vllm_max_soft_tokens", str(max_soft_tokens)]
    if enforce_eager:
        cmd.append("--vllm_enforce_eager")
    if revision:
        cmd += ["--revision", revision]
    _append_sources(cmd, source)

    runtime = (
        f"soft={max_soft_tokens or '-'} ctx={max_model_len} quant={quantization or '-'} "
        f"kv={kv_cache_dtype or '-'} tok={max_new_tokens}"
    )
    model = _adhoc_model(run_label, "vllm", model_path, runtime, notes)
    _print_run_plan(title=f"try-vllm {model_path}", cmd=cmd, models=[model], cfg=cfg, dry_run=dry_run)
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("try-transformers")
def try_transformers(
    model_path: str = typer.Argument(..., help="Repo HF ou chemin local safetensors."),
    label: str | None = typer.Option(None, help="Run label. Defaut derive du model_path."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int | None = typer.Option(20, help="Samples par source. Vide pour aucun cap."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    subsample_pct: float = typer.Option(100, help="Pourcentage du split."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    source: str | None = typer.Option(None, help="Sources separees par virgule, ex: docvqa,textvqa."),
    device: str = typer.Option("cuda", help="Device transformers."),
    max_new_tokens: int = typer.Option(32, help="Tokens generes."),
    temperature: float = typer.Option(1.0, help="Sampling temperature."),
    top_p: float = typer.Option(0.95, help="top_p."),
    top_k: int = typer.Option(64, help="top_k."),
    disable_energy_tracking: bool = typer.Option(False, "--no-energy-tracking", help="Desactive CodeCarbon."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Teste un repo HF avec le backend transformers sans l'ajouter au YAML."""
    cfg = _load_config()
    model_path = _normalize_hf_repo(model_path)
    run_label = label or _safe_label("tf", model_path)
    notes = "adhoc transformers"
    cmd = _direct_eval_base_cmd(
        backend="transformers",
        label=run_label,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        subsample_pct=subsample_pct,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        no_energy_tracking=disable_energy_tracking,
        notes=notes,
    )
    cmd += ["--model_path", model_path, "--device", device]
    _append_sources(cmd, source)

    model = _adhoc_model(run_label, "transformers", model_path, f"device={device} tok={max_new_tokens}", notes)
    _print_run_plan(title=f"try-transformers {model_path}", cmd=cmd, models=[model], cfg=cfg, dry_run=dry_run)
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("try-gguf")
def try_gguf(
    gguf_repo: str = typer.Argument(..., help="Repo HF ou dossier local du GGUF."),
    gguf_filename: str = typer.Argument(..., help="Nom ou glob du fichier .gguf."),
    label: str | None = typer.Option(None, help="Run label. Defaut derive du repo."),
    mmproj_repo: str | None = typer.Option(None, help="Repo HF ou dossier local du mmproj."),
    mmproj_filename: str | None = typer.Option(None, help="Nom ou glob du mmproj."),
    model_family: str = typer.Option("gemma4", help="gemma3, gemma4 ou llava."),
    n_gpu_layers: int = typer.Option(-1, help="-1 = toutes les couches GPU."),
    n_ctx: int = typer.Option(2048, help="Taille du contexte llama.cpp."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    limit: int | None = typer.Option(20, help="Samples par source. Vide pour aucun cap."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    subsample_pct: float = typer.Option(100, help="Pourcentage du split."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    source: str | None = typer.Option(None, help="Sources separees par virgule, ex: docvqa,textvqa."),
    max_new_tokens: int = typer.Option(32, help="Tokens generes."),
    temperature: float = typer.Option(1.0, help="Sampling temperature stockee dans config."),
    top_p: float = typer.Option(0.95, help="top_p stocke dans config."),
    top_k: int = typer.Option(64, help="top_k stocke dans config."),
    disable_energy_tracking: bool = typer.Option(False, "--no-energy-tracking", help="Desactive CodeCarbon."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Teste un GGUF via llama.cpp sans l'ajouter au YAML."""
    cfg = _load_config()
    gguf_repo = _normalize_hf_repo(gguf_repo)
    if mmproj_repo:
        mmproj_repo = _normalize_hf_repo(mmproj_repo)
    run_label = label or _safe_label("gguf", gguf_repo)
    notes = "adhoc llama_cpp"
    cmd = _direct_eval_base_cmd(
        backend="llama_cpp",
        label=run_label,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        limit=limit,
        country=country,
        split=split,
        subsample_pct=subsample_pct,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        no_energy_tracking=disable_energy_tracking,
        notes=notes,
    )
    cmd += [
        "--gguf_repo",
        gguf_repo,
        "--gguf_filename",
        gguf_filename,
        "--model_family",
        model_family,
        "--n_gpu_layers",
        str(n_gpu_layers),
        "--n_ctx",
        str(n_ctx),
    ]
    if mmproj_repo and mmproj_filename:
        cmd += ["--mmproj_repo", mmproj_repo, "--mmproj_filename", mmproj_filename]
    elif mmproj_repo or mmproj_filename:
        raise typer.BadParameter("--mmproj_repo et --mmproj_filename doivent etre fournis ensemble.")
    _append_sources(cmd, source)

    model = _adhoc_model(
        run_label,
        "llama_cpp",
        f"{gguf_repo}/{gguf_filename}",
        f"family={model_family} ctx={n_ctx} gpu_layers={n_gpu_layers} tok={max_new_tokens}",
        notes,
    )
    model["_model_ref"] = f"{gguf_repo}/{gguf_filename}"
    _print_run_plan(title=f"try-gguf {gguf_repo}/{gguf_filename}", cmd=cmd, models=[model], cfg=cfg, dry_run=dry_run)
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("final")
def final_run(
    target: str = typer.Argument(..., help="Label ou groupe a lancer sans limite."),
    dataset_hub: str | None = typer.Option(None, help="Dataset HF ou URL HF."),
    dataset_local: Path | None = typer.Option(None, help="Dataset local save_to_disk."),
    country: str = typer.Option(DEFAULT_COUNTRY, help="Code pays CodeCarbon."),
    split: str = typer.Option(DEFAULT_SPLIT, help="Split: test ou validation."),
    streaming_take: int | None = typer.Option(None, help="Samples HF lus en streaming."),
    streaming_per_source: int | None = typer.Option(None, help="Samples par source en streaming."),
    skip_existing: bool = typer.Option(False, help="Skip reports deja presents."),
    cleanup_cache: bool = typer.Option(False, help="Nettoie le cache HF apres run."),
    dry_run: bool = typer.Option(False, help="Afficher la commande sans l'executer."),
) -> None:
    """Lance un run final full split, sans limit_per_source."""
    cfg = _load_config()
    cmd = _final_cmd(
        target,
        dataset_hub=dataset_hub,
        dataset_local=dataset_local,
        country=country,
        split=split,
        streaming_take=streaming_take,
        streaming_per_source=streaming_per_source,
        skip_existing=skip_existing,
        cleanup_cache=cleanup_cache,
    )
    models = (
        _selected_models_for_target(cfg, group=target)
        if target in _groups(cfg)
        else _selected_models_for_target(cfg, label=target)
    )
    _print_run_plan(title=f"final {target}", cmd=cmd, models=models, cfg=cfg, dry_run=dry_run)
    if dry_run:
        return
    _run(cmd, dry_run, print_cmd=False)


@app.command("dry-run")
def dry_run_target(
    target: str = typer.Argument(..., help="Label ou groupe."),
    limit: int | None = typer.Option(20, help="Samples par source."),
) -> None:
    """Alias court pour afficher une commande de run."""
    cfg = _load_config()
    if target in _groups(cfg):
        cmd = _group_cmd(
            target,
            dataset_hub=None,
            dataset_local=None,
            limit=limit,
            country=DEFAULT_COUNTRY,
            split=DEFAULT_SPLIT,
            subsample_pct=100,
            streaming_take=None,
            streaming_per_source=None,
            skip_existing=False,
            cleanup_cache=False,
        )
    else:
        cmd = _label_cmd(
            target,
            dataset_hub=None,
            dataset_local=None,
            limit=limit,
            country=DEFAULT_COUNTRY,
            split=DEFAULT_SPLIT,
            subsample_pct=100,
            streaming_take=None,
            streaming_per_source=None,
            skip_existing=False,
            cleanup_cache=False,
        )
    _run(cmd, True)


@app.command()
def results(
    registry: Path = typer.Option(ROOT / "results" / "registry.csv", help="Registry CSV."),
    top: int = typer.Option(10, help="Nombre de lignes affichees."),
    sort: str = typer.Option("score_per_kwh", help="Colonne de tri."),
    backend: str | None = typer.Option(None, help="Filtre backend."),
) -> None:
    """Affiche les meilleures runs du registry."""
    allowed_sorts = {"score_per_kwh", "macro_avg", "samples_per_joule", "kwh_total"}
    if sort not in allowed_sorts:
        raise typer.BadParameter(
            f"Tri inconnu: {sort}. Choix: {', '.join(sorted(allowed_sorts))}"
        )
    if not registry.exists():
        raise typer.BadParameter(f"Registry introuvable: {registry}")
    with registry.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["backend"] = _infer_backend(row)
    if backend:
        rows = [row for row in rows if row["backend"] == backend]
    if not rows:
        console.print("[yellow]Aucune run trouvee.[/yellow]")
        return
    rows.sort(key=lambda row: _float_value(row, sort), reverse=(sort != "kwh_total"))

    table = Table(title=f"Top runs by {sort}")
    for column in (
        "run_label",
        "backend",
        "macro_avg",
        "score_per_kwh",
        "samples_per_joule",
        "kwh_total",
        "n_samples_total",
        "duration_s",
        "gpu_name",
    ):
        table.add_column(column, overflow="ellipsis", no_wrap=True)
    for row in rows[:top]:
        table.add_row(
            _short(row.get("run_label", ""), 34),
            row.get("backend", ""),
            _num(row, "macro_avg"),
            _num(row, "score_per_kwh"),
            _num(row, "samples_per_joule"),
            _num(row, "kwh_total"),
            row.get("n_samples_total", ""),
            _num(row, "duration_s"),
            _short(row.get("gpu_name", ""), 28),
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Verifie que la CLI peut tourner sur cette machine."""
    cfg = _load_config()
    checks = [
        ("python", platform.python_version(), True),
        ("bench_all.py", str(BENCH_ALL), BENCH_ALL.exists()),
        ("models.yaml", str(CONFIG_PATH), CONFIG_PATH.exists()),
        ("typer", "python module", _module_exists("typer")),
        ("rich", "python module", _module_exists("rich")),
        ("questionary", "python module", _module_exists("questionary")),
        ("yaml", "python module", _module_exists("yaml")),
        ("runpod dataset", str(RUNPOD_DATASET_LOCAL), _runpod_dataset_exists()),
        ("registry", str(ROOT / "results" / "registry.csv"), (ROOT / "results" / "registry.csv").exists()),
    ]
    table = Table(title="Bench CLI doctor")
    table.add_column("Check", style="bold")
    table.add_column("Detail")
    table.add_column("Status")
    for name, detail, status in checks:
        style = "green" if status else "yellow"
        table.add_row(name, detail, f"[{style}]{_ok(status)}[/{style}]")
    console.print(table)
    dupes = _duplicate_labels(cfg)
    missing_group_labels = [
        label
        for labels in _groups(cfg).values()
        for label in labels
        if label not in _models_by_label(cfg)
    ]
    console.print(
        Panel(
            f"{len(cfg.get('models', []))} modeles, {len(_groups(cfg))} groupes. "
            f"Dataset par defaut: {DEFAULT_DATASET_HUB}\n"
            f"Labels uniques: {len(_models_by_label(cfg))}\n"
            f"Labels dupliques: {', '.join(sorted(dupes)) if dupes else 'aucun'}\n"
            f"Labels de groupe manquants: {', '.join(missing_group_labels) if missing_group_labels else 'aucun'}",
            title="Config",
        )
    )


@app.command()
def menu() -> None:
    """Menu interactif avec fleches/espace/entree."""
    _require_questionary()
    cfg = _load_config()
    action = questionary.select(
        "Que veux-tu lancer ?",
        choices=[
            _choice("Smoke rapide", "smoke"),
            _choice("Run modele", "run"),
            _choice("Run plusieurs modeles", "batch"),
            _choice("Run groupe", "group"),
            _choice("Final full split", "final"),
            _choice("Voir resultats", "results"),
            _choice("Voir sources/metrics", "sources"),
            _choice("Expliquer le benchmark", "explain"),
            _choice("Doctor", "doctor"),
        ],
    ).ask()
    if action is None:
        raise typer.Exit(130)
    if action == "results":
        results()
        return
    if action == "doctor":
        doctor()
        return
    if action == "sources":
        sources()
        return
    if action == "explain":
        explain()
        return

    dataset_choice = questionary.select(
        "Dataset",
        choices=[
            _choice("Auto Runpod local sinon HF resilient", "auto"),
            _choice("HF resilient par defaut", "default_hf"),
            _choice("Autre lien/repo HF", "custom_hf"),
            _choice("Chemin local", "local"),
        ],
    ).ask()
    if dataset_choice is None:
        raise typer.Exit(130)

    dataset_hub = None
    dataset_local = None
    if dataset_choice == "custom_hf":
        value = questionary.text("Lien ou repo id HF dataset:").ask()
        if not value:
            raise typer.Exit(130)
        dataset_hub = _normalize_hf_dataset(value)
    elif dataset_choice == "default_hf":
        dataset_hub = DEFAULT_DATASET_HUB
    elif dataset_choice == "local":
        value = questionary.path("Chemin dataset local:").ask()
        if not value:
            raise typer.Exit(130)
        dataset_local = Path(value)

    dry_run = bool(questionary.confirm("Dry-run seulement ?", default=True).ask())
    skip_existing = bool(questionary.confirm("Skip reports existants ?", default=True).ask())
    cleanup_cache = bool(questionary.confirm("Cleanup cache HF apres chaque modele ?", default=False).ask())

    if action == "smoke":
        backend = questionary.select(
            "Backend",
            choices=[_choice("vLLM", "vllm"), _choice("llama.cpp", "llama_cpp")],
        ).ask()
        if not backend:
            raise typer.Exit(130)
        cmd = _smoke_cmd(
            str(backend),
            dataset_hub=dataset_hub,
            dataset_local=dataset_local,
            limit=2,
            country=DEFAULT_COUNTRY,
            split=DEFAULT_SPLIT,
            streaming_take=64,
            streaming_per_source=None,
            skip_existing=skip_existing,
            cleanup_cache=cleanup_cache,
        )
        models = _selected_models_for_target(
            cfg,
            group="vllm_smoke" if backend == "vllm" else None,
            label="gemma4-e4b-gguf-q4km" if backend == "llama_cpp" else None,
        )
        _print_run_plan(title=f"menu smoke {backend}", cmd=cmd, models=models, cfg=cfg, dry_run=dry_run)
        if not dry_run:
            _run(cmd, dry_run, print_cmd=False)
        return

    if action == "run":
        label = questionary.select(
            "Modele",
            choices=list(_models_by_label(cfg).keys()),
            use_shortcuts=True,
        ).ask()
        if not label:
            raise typer.Exit(130)
        limit = questionary.text("Limit per source", default="20").ask()
        cmd = _label_cmd(
            str(label),
            dataset_hub=dataset_hub,
            dataset_local=dataset_local,
            limit=_parse_limit(limit),
            country=DEFAULT_COUNTRY,
            split=DEFAULT_SPLIT,
            subsample_pct=100,
            streaming_take=None,
            streaming_per_source=None,
            skip_existing=skip_existing,
            cleanup_cache=cleanup_cache,
        )
        _print_run_plan(
            title=f"menu run {label}",
            cmd=cmd,
            models=_selected_models_for_target(cfg, label=str(label)),
            cfg=cfg,
            dry_run=dry_run,
        )
        if not dry_run:
            _run(cmd, dry_run, print_cmd=False)
        return

    if action == "batch":
        labels = questionary.checkbox(
            "Modeles a lancer (espace pour selectionner)",
            choices=list(_models_by_label(cfg).keys()),
            use_shortcuts=True,
        ).ask()
        if not labels:
            raise typer.Exit(130)
        limit = questionary.text("Limit per source", default="20").ask()
        cmd = _labels_cmd(
            [str(label) for label in labels],
            dataset_hub=dataset_hub,
            dataset_local=dataset_local,
            limit=_parse_limit(limit),
            country=DEFAULT_COUNTRY,
            split=DEFAULT_SPLIT,
            subsample_pct=100,
            streaming_take=None,
            streaming_per_source=None,
            skip_existing=skip_existing,
            cleanup_cache=cleanup_cache,
        )
        str_labels = [str(label) for label in labels]
        _print_run_plan(
            title=f"menu batch {len(str_labels)} labels",
            cmd=cmd,
            models=_selected_models_for_target(cfg, labels=str_labels),
            cfg=cfg,
            dry_run=dry_run,
        )
        if not dry_run:
            _run(cmd, dry_run, print_cmd=False)
        return

    group_name = questionary.select(
        "Groupe",
        choices=list(_groups(cfg).keys()),
        use_shortcuts=True,
    ).ask()
    if not group_name:
        raise typer.Exit(130)
    if action == "group":
        limit = questionary.text("Limit per source", default="20").ask()
        cmd = _group_cmd(
            str(group_name),
            dataset_hub=dataset_hub,
            dataset_local=dataset_local,
            limit=_parse_limit(limit),
            country=DEFAULT_COUNTRY,
            split=DEFAULT_SPLIT,
            subsample_pct=100,
            streaming_take=None,
            streaming_per_source=None,
            skip_existing=skip_existing,
            cleanup_cache=cleanup_cache,
        )
        _print_run_plan(
            title=f"menu group {group_name}",
            cmd=cmd,
            models=_selected_models_for_target(cfg, group=str(group_name)),
            cfg=cfg,
            dry_run=dry_run,
        )
        if not dry_run:
            _run(cmd, dry_run, print_cmd=False)
    elif action == "final":
        cmd = _final_cmd(
            str(group_name),
            dataset_hub=dataset_hub,
            dataset_local=dataset_local,
            country=DEFAULT_COUNTRY,
            split=DEFAULT_SPLIT,
            streaming_take=None,
            streaming_per_source=None,
            skip_existing=skip_existing,
            cleanup_cache=cleanup_cache,
        )
        _print_run_plan(
            title=f"menu final {group_name}",
            cmd=cmd,
            models=_selected_models_for_target(cfg, group=str(group_name)),
            cfg=cfg,
            dry_run=dry_run,
        )
        if not dry_run:
            _run(cmd, dry_run, print_cmd=False)


if __name__ == "__main__":
    app()
