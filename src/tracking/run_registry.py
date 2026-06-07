"""
run_registry.py — registry append-only de toutes les runs benchmark.

Objectif : avoir UNE source de vérité pour comparer 10/20/100 runs sur la
durée du challenge — chaque ligne = 1 run, colonnes = scores + énergie +
hardware + commit. CSV pour pouvoir l'ouvrir dans Excel / pandas / un
dashboard sans setup.

Workflow :
    from run_registry import RegistryEntry, append_to_registry

    entry = RegistryEntry.from_benchmark(
        run_label="qat-int4-v2",
        bench_result=result,            # benchmark.BenchmarkResult
        energy_snapshot=tracker.snapshot,
        config={"quant": "int4", "subsample_pct": 10, "seed": 42},
    )
    append_to_registry(entry, csv_path="results/registry.csv")

Le CSV est créé avec en-tête au premier append, puis seules les lignes
sont ajoutées ensuite. Compatible reprise après crash.

Bonus : `dump_run_artifacts()` sérialise une run complète en artefacts
côte-à-côte : JSON, énergie par source, résumé texte, et tableau jury-like.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.tracking.energy_tracker import EnergySnapshot, derive_efficiency_metrics
from src.validation.benchmark import (
    LEADERBOARD_COLUMNS,
    OFFICIAL_BASE_MODEL,
    OFFICIAL_INTELLIGENCE_LAB_ECE,
    OFFICIAL_TOP_ROUND1,
)


# ---------------------------------------------------------------------------
# Schéma RegistryEntry
# ---------------------------------------------------------------------------
# Ordre des colonnes du CSV global. Toutes nouvelles colonnes vont à la fin
# pour ne pas casser les CSVs existants → schema-evolution friendly.
REGISTRY_COLUMNS: list[str] = [
    # --- Identifiants ---
    "timestamp_iso",
    "run_id",
    "run_label",
    "git_commit",
    # --- Modèle / config ---
    "model_id",
    "split",
    "subsample_pct",
    "subsample_seed",
    "n_samples_total",
    "n_sources",
    "config_json",
    # --- Scores benchmark ---
    "macro_avg",
    "micro_avg",
    "per_source_json",
    # --- Énergie (CodeCarbon) ---
    "energy_backend",
    "kwh_total",
    "kwh_cpu",
    "kwh_gpu",
    "kwh_ram",
    "co2_g",
    "country_iso",
    "duration_s",
    # --- KPIs efficacité ---
    "joules_per_sample",
    "samples_per_joule",
    "wh_per_sample",
    "co2_mg_per_sample",
    "score_per_kwh",
    # --- Hardware ---
    "hostname",
    "gpu_name",
    "gpu_count",
    "ram_total_gb",
    # --- Notes libres ---
    "notes",
    # --- Runtime stats optionnelles ---
    "adaptive_trigger_count",
    "adaptive_max_power_w",
    "adaptive_throttle_remaining",
    "runtime_stats_json",
    # --- Colonnes jury-like ---
    "official_energy_consumption",
    "official_document_analysis",
    "official_image_understanding",
    "official_mean",
    "official_mean_recovery",
    "official_delta_mean_vs_intelligence_lab_ece",
    "official_missing_sources_json",
    "leaderboard_like_json",
]


@dataclass
class RegistryEntry:
    """Une ligne dans le registry global. Tout au plat, prêt à dumper."""
    timestamp_iso: str = ""
    run_id: str = ""
    run_label: str = ""
    git_commit: str | None = None

    model_id: str = ""
    split: str = ""
    subsample_pct: float = 100.0
    subsample_seed: int | None = None
    n_samples_total: int = 0
    n_sources: int = 0
    config_json: str = "{}"

    macro_avg: float | None = None
    micro_avg: float | None = None
    per_source_json: str = "{}"

    energy_backend: str = "n/a"
    kwh_total: float = 0.0
    kwh_cpu: float = 0.0
    kwh_gpu: float = 0.0
    kwh_ram: float = 0.0
    co2_g: float = 0.0
    country_iso: str | None = None
    duration_s: float = 0.0

    joules_per_sample: float = 0.0
    samples_per_joule: float = 0.0
    wh_per_sample: float = 0.0
    co2_mg_per_sample: float = 0.0
    score_per_kwh: float | None = None

    hostname: str = ""
    gpu_name: str = ""
    gpu_count: int = 0
    ram_total_gb: float | None = None

    notes: str = ""
    adaptive_trigger_count: int | None = None
    adaptive_max_power_w: float | None = None
    adaptive_throttle_remaining: int | None = None
    runtime_stats_json: str = "{}"
    official_energy_consumption: float | None = None
    official_document_analysis: float | None = None
    official_image_understanding: float | None = None
    official_mean: float | None = None
    official_mean_recovery: float | None = None
    official_delta_mean_vs_intelligence_lab_ece: float | None = None
    official_missing_sources_json: str = "{}"
    leaderboard_like_json: str = "[]"

    # ---- Construction depuis les objets benchmark + énergie ----
    @classmethod
    def from_benchmark(
        cls,
        run_label: str,
        bench_result,                # benchmark.BenchmarkResult (duck-typed)
        energy_snapshot: EnergySnapshot,
        config: dict[str, Any] | None = None,
        notes: str = "",
        subsample_pct: float = 100.0,
        subsample_seed: int | None = None,
    ) -> "RegistryEntry":
        cfg = dict(config or {})
        per_source = {
            r.source: {
                "score": r.score, "metric": r.metric,
                "n_scored": r.n_scored, "n_samples": r.n_samples,
            }
            for r in (bench_result.per_source or [])
        }
        # Score utilisé pour score_per_kwh : macro avg
        snap = energy_snapshot
        snap_score = bench_result.macro_avg
        snap.score = snap_score
        snap.n_samples = sum(r.n_samples for r in (bench_result.per_source or []))
        eff = derive_efficiency_metrics(snap)

        gpus = snap.hardware.get("gpus") or []
        gpu_name = gpus[0]["name"] if gpus else ""
        runtime_stats = cfg.get("runtime_stats") or {}
        adaptive_stats = runtime_stats.get("adaptive_soft_tokens") or {}
        official_like = _official_like_dict(getattr(bench_result, "official_like", None))
        leaderboard = _leaderboard_rows(
            run_label=run_label,
            official_like=official_like,
            joules_per_sample=eff.get("joules_per_sample", 0.0),
        )
        missing_sources = {
            "document": official_like.get("document_sources_missing") or [],
            "image": official_like.get("image_sources_missing") or [],
        }

        return cls(
            timestamp_iso=datetime.fromtimestamp(snap.started_at, tz=timezone.utc)
                                    .isoformat(timespec="seconds"),
            run_id=snap.run_id,
            run_label=run_label,
            git_commit=snap.hardware.get("git_commit"),

            model_id=getattr(bench_result, "model_id", ""),
            split=getattr(bench_result, "split", ""),
            subsample_pct=float(subsample_pct),
            subsample_seed=subsample_seed,
            n_samples_total=snap.n_samples,
            n_sources=len(per_source),
            config_json=json.dumps(cfg, ensure_ascii=False, sort_keys=True),

            macro_avg=bench_result.macro_avg,
            micro_avg=bench_result.micro_avg,
            per_source_json=json.dumps(per_source, ensure_ascii=False),

            energy_backend=snap.backend,
            kwh_total=snap.kwh,
            kwh_cpu=snap.kwh_cpu,
            kwh_gpu=snap.kwh_gpu,
            kwh_ram=snap.kwh_ram,
            co2_g=snap.co2_g,
            country_iso=snap.country_iso,
            duration_s=snap.duration_s,

            joules_per_sample=eff.get("joules_per_sample", 0.0),
            samples_per_joule=eff.get("samples_per_joule", 0.0),
            wh_per_sample=eff.get("wh_per_sample", 0.0),
            co2_mg_per_sample=eff.get("co2_mg_per_sample", 0.0),
            score_per_kwh=eff.get("score_per_kwh"),

            hostname=snap.hardware.get("hostname", ""),
            gpu_name=gpu_name,
            gpu_count=len(gpus),
            ram_total_gb=(snap.hardware.get("cpu") or {}).get("ram_total_gb"),

            notes=notes,
            adaptive_trigger_count=adaptive_stats.get("trigger_count"),
            adaptive_max_power_w=adaptive_stats.get("max_power_w"),
            adaptive_throttle_remaining=adaptive_stats.get("throttle_remaining"),
            runtime_stats_json=json.dumps(runtime_stats, ensure_ascii=False, sort_keys=True),
            official_energy_consumption=eff.get("joules_per_sample", 0.0),
            official_document_analysis=official_like.get("document_analysis"),
            official_image_understanding=official_like.get("image_understanding"),
            official_mean=official_like.get("mean"),
            official_mean_recovery=official_like.get("mean_recovery"),
            official_delta_mean_vs_intelligence_lab_ece=official_like.get(
                "delta_mean_vs_intelligence_lab_ece"
            ),
            official_missing_sources_json=json.dumps(missing_sources, ensure_ascii=False, sort_keys=True),
            leaderboard_like_json=json.dumps(leaderboard, ensure_ascii=False),
        )

    # ---- Sérialisation -----------------------------------------------------
    def to_csv_row(self) -> list[str]:
        d = asdict(self)
        return [_csv_safe(d.get(c)) for c in REGISTRY_COLUMNS]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _csv_safe(v: Any) -> str:
    """Convertit n'importe quoi en chaîne CSV-safe (None → vide)."""
    if v is None:
        return ""
    if isinstance(v, float):
        # 6 décimales suffit pour kWh ; l'idée n'est pas la précision absolue
        # mais d'avoir un CSV lisible.
        return f"{v:.6g}"
    return str(v)


# ---------------------------------------------------------------------------
# Append-only writer
# ---------------------------------------------------------------------------
def append_to_registry(
    entry: RegistryEntry,
    csv_path: str | Path = "results/registry.csv",
) -> Path:
    """Append `entry` au CSV global. Crée le fichier + header si absent."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    columns = _ensure_registry_columns(p)
    is_new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(columns)
        w.writerow(_entry_row_for_columns(entry, columns))
    return p


def append_many(
    entries: Iterable[RegistryEntry],
    csv_path: str | Path = "results/registry.csv",
) -> Path:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns = _ensure_registry_columns(p)
    is_new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(columns)
        for e in entries:
            w.writerow(_entry_row_for_columns(e, columns))
    return p


# ---------------------------------------------------------------------------
# Dump détaillé d'une run (en plus du registry)
# ---------------------------------------------------------------------------
def dump_run_artifacts(
    run_label: str,
    bench_result,
    energy_snapshot: EnergySnapshot,
    per_source_snapshots: list[EnergySnapshot] | None = None,
    out_dir: str | Path = "results/runs",
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Sérialise une run en 3 artefacts à côté du registry global :

      - `<label>_<ts>.json`         : tout (scores + énergie + config)
      - `<label>_<ts>_energy.csv`   : énergie par-source (1 ligne / source)
      - `<label>_<ts>_summary.txt`  : résumé human-readable 1 page

    Retourne un dict des paths créés.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromtimestamp(energy_snapshot.started_at, tz=timezone.utc) \
                  .strftime("%Y%m%dT%H%M%SZ")
    base = out / f"{_slug(run_label)}_{ts}"

    # ------ JSON global ------
    payload = {
        "label": run_label,
        "config": config or {},
        "benchmark": json.loads(bench_result.to_json()) if hasattr(bench_result, "to_json") else None,
        "energy": energy_snapshot.to_dict(),
        "efficiency_kpi": derive_efficiency_metrics(energy_snapshot),
        "per_source_energy": [s.to_dict() for s in (per_source_snapshots or [])],
    }
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # ------ CSV par-source ------
    csv_path = Path(str(base) + "_energy.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source_or_task", "n_samples", "duration_s", "kwh_total",
            "kwh_cpu", "kwh_gpu", "kwh_ram", "co2_g",
            "joules_per_sample", "samples_per_joule",
        ])
        for s in (per_source_snapshots or []):
            eff = derive_efficiency_metrics(s)
            w.writerow([
                s.task, s.n_samples, f"{s.duration_s:.3f}", f"{s.kwh:.6g}",
                f"{s.kwh_cpu:.6g}", f"{s.kwh_gpu:.6g}", f"{s.kwh_ram:.6g}",
                f"{s.co2_g:.6g}",
                f"{eff['joules_per_sample']:.6g}",
                f"{eff['samples_per_joule']:.6g}",
            ])

    # ------ Résumé texte ------
    txt_path = Path(str(base) + "_summary.txt")
    snap = energy_snapshot
    eff = derive_efficiency_metrics(snap)
    lines = [
        f"=== {run_label}  ({ts}) ===",
        f"model_id   : {getattr(bench_result, 'model_id', '?')}",
        f"split      : {getattr(bench_result, 'split', '?')}",
        f"macro_avg  : {bench_result.macro_avg:.4f}",
        f"micro_avg  : {bench_result.micro_avg:.4f}",
        f"n_samples  : {snap.n_samples}",
        f"duration_s : {snap.duration_s:.1f}",
        "",
        f"kWh total  : {snap.kwh:.6g}    (cpu={snap.kwh_cpu:.4g}  gpu={snap.kwh_gpu:.4g}  ram={snap.kwh_ram:.4g})",
        f"co2_g      : {snap.co2_g:.4g}    country={snap.country_iso or '?'}",
        f"backend    : {snap.backend}",
        "",
        f"J/sample   : {eff['joules_per_sample']:.4g}",
        f"sample/J   : {eff['samples_per_joule']:.4g}",
        f"Wh/sample  : {eff['wh_per_sample']:.4g}",
    ]
    if "score_per_kwh" in eff:
        lines.append(f"score/kWh  : {eff['score_per_kwh']:.4g}")

    official_like = _official_like_dict(getattr(bench_result, "official_like", None))
    paths = {"json": json_path, "energy_csv": csv_path, "summary": txt_path}
    if official_like:
        leaderboard = _leaderboard_rows(run_label, official_like, eff["joules_per_sample"])
        leaderboard_csv_path = Path(str(base) + "_leaderboard_like.csv")
        with leaderboard_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
            w.writeheader()
            for row in leaderboard:
                w.writerow({column: row.get(column) for column in LEADERBOARD_COLUMNS})

        leaderboard_md_path = Path(str(base) + "_leaderboard_like.md")
        leaderboard_md_path.write_text("\n".join(_leaderboard_markdown(leaderboard)) + "\n")

        lines.extend(
            [
                "",
                "Format jury",
                f"document_analysis : {_fmt_optional(official_like.get('document_analysis'))}",
                f"image_understanding: {_fmt_optional(official_like.get('image_understanding'))}",
                f"mean               : {_fmt_optional(official_like.get('mean'))}",
                f"mean_recovery      : {_fmt_optional(official_like.get('mean_recovery'))}",
            ]
        )
        paths["leaderboard_csv"] = leaderboard_csv_path
        paths["leaderboard_md"] = leaderboard_md_path
    txt_path.write_text("\n".join(lines))

    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    return "".join(c if c in keep else "-" for c in s.lower())[:60].strip("-") or "run"


def _ensure_registry_columns(path: Path) -> list[str]:
    """Migre le header CSV si des colonnes append-only ont ete ajoutees."""
    if not path.exists() or path.stat().st_size == 0:
        return list(REGISTRY_COLUMNS)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        rows = list(reader)

    columns = list(existing)
    changed = False
    for column in REGISTRY_COLUMNS:
        if column not in columns:
            columns.append(column)
            changed = True

    if not changed:
        return columns

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    os.replace(tmp, path)
    return columns


def _entry_row_for_columns(entry: RegistryEntry, columns: list[str]) -> list[str]:
    d = entry.to_dict()
    return [_csv_safe(d.get(c)) for c in columns]


def _official_like_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return asdict(value)


def _leaderboard_rows(
    run_label: str,
    official_like: dict[str, Any],
    joules_per_sample: float,
) -> list[dict[str, Any]]:
    return [
        {"team_name": "base_model", **OFFICIAL_BASE_MODEL},
        {"team_name": "Top model from Round 1", **OFFICIAL_TOP_ROUND1},
        {"team_name": "Intelligence Lab ECE", **OFFICIAL_INTELLIGENCE_LAB_ECE},
        {
            "team_name": run_label,
            "energy_consumption": joules_per_sample,
            "document_analysis": official_like.get("document_analysis"),
            "image_understanding": official_like.get("image_understanding"),
            "mean": official_like.get("mean"),
            "mean_recovery": official_like.get("mean_recovery"),
        },
    ]


def _fmt_optional(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _leaderboard_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| team_name | energy_consumption | document_analysis | image_understanding | mean | mean_recovery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_leaderboard_value(row.get(column)) for column in LEADERBOARD_COLUMNS)
            + " |"
        )
    return lines


def _leaderboard_value(value: Any) -> str:
    return "" if value is None else str(value)


__all__ = [
    "REGISTRY_COLUMNS",
    "RegistryEntry",
    "append_to_registry",
    "append_many",
    "dump_run_artifacts",
]
