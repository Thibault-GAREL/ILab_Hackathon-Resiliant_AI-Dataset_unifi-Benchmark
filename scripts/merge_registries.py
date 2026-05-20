#!/usr/bin/env python
"""Fusionne plusieurs registry.csv en un CSV comparable."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _infer_backend(row: dict[str, str]) -> str:
    try:
        cfg: dict[str, Any] = json.loads(row.get("config_json") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    if cfg.get("backend"):
        return str(cfg["backend"])
    label = row.get("run_label", "").lower()
    if "gguf" in label:
        return "llama_cpp"
    if "vllm" in label:
        return "vllm"
    return "transformers"


def _infer_family(row: dict[str, str]) -> str:
    label = row.get("run_label", "").lower()
    if "vllm" in label:
        return "vllm"
    if "gguf" in label:
        return "gguf"
    if "awq" in label:
        return "awq"
    if "bf16" in label:
        return "bf16"
    return "other"


def _read_registry(path: Path, source_name: str) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["registry_source"] = source_name
        row["registry_path"] = str(path)
        row["backend_inferred"] = _infer_backend(row)
        row["family_inferred"] = _infer_family(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("registries", nargs="+", type=Path)
    ap.add_argument("--name", action="append", default=[],
                    help="Nom lisible pour chaque registry, dans le même ordre.")
    ap.add_argument("--out", type=Path, default=Path("results/combined_registry.csv"))
    args = ap.parse_args()

    if args.name and len(args.name) != len(args.registries):
        raise SystemExit("--name doit être répété autant de fois que les registries.")

    all_rows: list[dict[str, str]] = []
    for idx, path in enumerate(args.registries):
        name = args.name[idx] if args.name else path.stem
        all_rows.extend(_read_registry(path.expanduser(), name))

    if not all_rows:
        raise SystemExit("Aucune ligne trouvée.")

    fieldnames: list[str] = []
    for key in ("registry_source", "backend_inferred", "family_inferred", "registry_path"):
        if key not in fieldnames:
            fieldnames.append(key)
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[merge] {len(all_rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
