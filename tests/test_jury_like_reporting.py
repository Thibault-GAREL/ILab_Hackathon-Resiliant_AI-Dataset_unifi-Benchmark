from __future__ import annotations

import csv
import json
from types import SimpleNamespace


def test_compute_official_like_scores_matches_jury_formula() -> None:
    from src.validation.benchmark import (
        DOCUMENT_ANALYSIS_SOURCES,
        IMAGE_UNDERSTANDING_SOURCES,
        OFFICIAL_BASE_MODEL,
        PerSourceResult,
        compute_official_like_scores,
    )

    per_source = [
        PerSourceResult(source, 1, 1, 0.8, "test", {})
        for source in DOCUMENT_ANALYSIS_SOURCES
    ] + [
        PerSourceResult(source, 1, 1, 0.6, "test", {})
        for source in IMAGE_UNDERSTANDING_SOURCES
    ]

    official = compute_official_like_scores(per_source)

    assert official.document_analysis == 0.8
    assert official.image_understanding == 0.6
    assert official.mean == 0.7
    assert official.mean_recovery == 0.7 / OFFICIAL_BASE_MODEL["mean"] * 100.0
    assert official.document_sources_missing == []
    assert official.image_sources_missing == []


def test_dump_run_artifacts_writes_jury_like_table(tmp_path) -> None:
    from src.tracking.energy_tracker import EnergySnapshot
    from src.tracking.run_registry import dump_run_artifacts
    from src.validation.benchmark import (
        DOCUMENT_ANALYSIS_SOURCES,
        IMAGE_UNDERSTANDING_SOURCES,
        PerSourceResult,
        compute_official_like_scores,
    )

    per_source = [
        PerSourceResult(source, 1, 1, 0.8, "test", {})
        for source in DOCUMENT_ANALYSIS_SOURCES
    ] + [
        PerSourceResult(source, 1, 1, 0.6, "test", {})
        for source in IMAGE_UNDERSTANDING_SOURCES
    ]
    bench_result = SimpleNamespace(
        model_id="Mathos34400/resilient-challenge-image-to-text/gemma-4-q4_k_m.gguf",
        split="test",
        macro_avg=0.7,
        micro_avg=0.7,
        per_source=per_source,
        official_like=compute_official_like_scores(per_source),
    )
    snap = EnergySnapshot(
        run_id="run123",
        project="jury",
        task="all_sources",
        started_at=1_717_000_000.0,
        duration_s=12.0,
        kwh=0.001,
        kwh_cpu=0.0002,
        kwh_gpu=0.0007,
        kwh_ram=0.0001,
        co2_g=0.2,
        country_iso="CZE",
        hardware={"hostname": "l4-test", "gpus": [{"name": "NVIDIA L4"}], "cpu": {}},
        backend="codecarbon",
        n_samples=12,
    )

    paths = dump_run_artifacts("local Mathos L4 test", bench_result, snap, out_dir=tmp_path)

    assert "leaderboard_csv" in paths
    with paths["leaderboard_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["team_name"] == "base_model"
    assert rows[1]["team_name"] == "Top model from Round 1"
    assert rows[2]["team_name"] == "Intelligence Lab ECE"
    assert rows[3]["team_name"] == "local Mathos L4 test"
    assert rows[3]["document_analysis"] == "0.8"
    assert rows[3]["image_understanding"] == "0.6"
    assert rows[3]["mean"] == "0.7"


def test_registry_migrates_old_header_for_jury_columns(tmp_path) -> None:
    from src.tracking.energy_tracker import EnergySnapshot
    from src.tracking.run_registry import REGISTRY_COLUMNS, RegistryEntry, append_to_registry
    from src.validation.benchmark import PerSourceResult, compute_official_like_scores

    registry = tmp_path / "registry.csv"
    registry.write_text("timestamp_iso,run_id,run_label,macro_avg\nold,oldrun,oldlabel,0.1\n")

    per_source = [
        PerSourceResult("docvqa", 1, 1, 0.8, "test", {}),
        PerSourceResult("vqav2", 1, 1, 0.6, "test", {}),
    ]
    bench_result = SimpleNamespace(
        model_id="model",
        split="test",
        macro_avg=0.7,
        micro_avg=0.7,
        per_source=per_source,
        official_like=compute_official_like_scores(per_source),
    )
    snap = EnergySnapshot(
        run_id="newrun",
        project="jury",
        task="all_sources",
        started_at=1_717_000_000.0,
        duration_s=12.0,
        kwh=0.001,
        kwh_cpu=0.0002,
        kwh_gpu=0.0007,
        kwh_ram=0.0001,
        co2_g=0.2,
        country_iso="CZE",
        hardware={"hostname": "l4-test", "gpus": [{"name": "NVIDIA L4"}], "cpu": {}},
        backend="codecarbon",
        n_samples=2,
    )

    entry = RegistryEntry.from_benchmark("jury-run", bench_result, snap)
    append_to_registry(entry, registry)

    with registry.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["run_id"] == "oldrun"
    assert rows[1]["run_id"] == "newrun"
    assert "official_mean" in rows[1]
    assert json.loads(rows[1]["leaderboard_like_json"])[-1]["team_name"] == "jury-run"
    assert set(REGISTRY_COLUMNS).issubset(rows[1].keys())
