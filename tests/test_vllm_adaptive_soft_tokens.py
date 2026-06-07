from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_bench_all():
    spec = importlib.util.spec_from_file_location("bench_all", ROOT / "scripts" / "bench_all.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_all_passes_vllm_adaptive_soft_token_args() -> None:
    bench_all = load_bench_all()
    model = {
        "label": "adaptive-vllm",
        "backend": "vllm",
        "model_path": "google/gemma-4-E4B-it",
        "vllm_max_soft_tokens": 140,
        "vllm_adaptive_soft_tokens": "nvml_peak_v1",
        "vllm_adaptive_peak_power_w": 65.0,
        "vllm_adaptive_throttled_soft_tokens": 70,
        "vllm_adaptive_cooldown_requests": 3,
    }
    defaults = {
        "split": "test",
        "subsample_pct": 100,
        "country_iso": "CZE",
        "dataset_hub": "ECE-ILAB/resilient-ai-unified",
    }
    args = Namespace(
        split=None,
        subsample_pct=None,
        country_iso=None,
        dataset_local=None,
        dataset_hub=None,
        dataset_streaming_per_source=2,
        dataset_streaming_take=None,
        limit_per_source=2,
    )

    cmd = bench_all._build_eval_cmd(model, defaults, args)

    assert cmd[cmd.index("--vllm_adaptive_soft_tokens") + 1] == "nvml_peak_v1"
    assert cmd[cmd.index("--vllm_adaptive_peak_power_w") + 1] == "65.0"
    assert cmd[cmd.index("--vllm_adaptive_throttled_soft_tokens") + 1] == "70"
    assert cmd[cmd.index("--vllm_adaptive_cooldown_requests") + 1] == "3"


def test_adaptive_state_reports_runtime_stats() -> None:
    from src.models.vllm_backend import _AdaptiveSoftTokenState

    state = _AdaptiveSoftTokenState(
        default_soft_tokens=140,
        throttled_soft_tokens=70,
        peak_power_w=65.0,
        peak_active_power_w=None,
        baseline_power_w=0.0,
        cooldown_requests=2,
    )

    assert state.next_soft_tokens(None) == 140
    state.observe(66.5)

    assert state.next_soft_tokens(None) == 70
    stats = state.runtime_stats()
    assert stats["trigger_count"] == 1
    assert stats["max_power_w"] == 66.5
    assert stats["throttle_remaining"] == 1


def test_registry_promotes_adaptive_runtime_stats_to_columns() -> None:
    from src.tracking.energy_tracker import EnergySnapshot
    from src.tracking.run_registry import REGISTRY_COLUMNS, RegistryEntry

    bench_result = SimpleNamespace(
        model_id="google/gemma-4-E4B-it",
        split="test",
        macro_avg=0.5,
        micro_avg=0.4,
        per_source=[
            SimpleNamespace(
                source="vqav2",
                score=0.5,
                metric="relaxed_accuracy",
                n_scored=1,
                n_samples=1,
            )
        ],
    )
    energy_snapshot = EnergySnapshot(
        run_id="run123",
        project="adaptive",
        task="all_sources",
        started_at=1_717_000_000.0,
        duration_s=12.0,
        kwh=0.001,
        kwh_cpu=0.0002,
        kwh_gpu=0.0007,
        kwh_ram=0.0001,
        co2_g=0.2,
        country_iso="CZE",
        hardware={
            "hostname": "l4-test",
            "git_commit": "abc123",
            "gpus": [{"name": "NVIDIA L4"}],
            "cpu": {"ram_total_gb": 62.0},
        },
        backend="codecarbon",
    )
    config = {
        "runtime_stats": {
            "adaptive_soft_tokens": {
                "trigger_count": 2,
                "max_power_w": 66.5,
                "throttle_remaining": 1,
            }
        }
    }

    entry = RegistryEntry.from_benchmark(
        run_label="adaptive-vllm",
        bench_result=bench_result,
        energy_snapshot=energy_snapshot,
        config=config,
    )
    row = dict(zip(REGISTRY_COLUMNS, entry.to_csv_row()))

    assert entry.adaptive_trigger_count == 2
    assert entry.adaptive_max_power_w == 66.5
    assert entry.adaptive_throttle_remaining == 1
    assert row["adaptive_trigger_count"] == "2"
    assert json.loads(row["runtime_stats_json"]) == config["runtime_stats"]
