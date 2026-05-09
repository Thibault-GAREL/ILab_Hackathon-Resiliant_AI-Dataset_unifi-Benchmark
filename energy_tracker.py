"""
energy_tracker.py — wrapper CodeCarbon pour le challenge Resilient AI.

Pourquoi : le ranking PEReN se base sur la conso énergie (CodeCarbon)
*plus* le seuil 80% sur les 7 benchmarks. L'énergie domine la priorisation —
on ne peut donc PAS se contenter d'un timing / wall_time.

Ce module expose :
  - `EnergyTracker` : context-manager léger qui wrap codecarbon.EmissionsTracker
    et collecte en plus les infos hardware (GPU NVML / CPU / RAM) qu'on perd
    sinon. Idempotent : si codecarbon n'est pas installé, on dégrade en
    mesures wall_time-only sans bloquer la run.
  - `EnergySnapshot` : dataclass sérialisable qui contient tout ce dont on
    a besoin pour comparer deux runs.
  - `aggregate_snapshots()` : combine plusieurs snapshots (par-source) en
    un seul snapshot total — utile en fin de run.
  - `derive_efficiency_metrics()` : calcule samples/J, J/sample, score/J
    qui sont les vrais KPIs challenge.

Usage minimal :

    from energy_tracker import EnergyTracker

    with EnergyTracker(project="bench-int4", task="layer1_full") as tr:
        run_my_eval()
    print(tr.snapshot.kwh, tr.snapshot.co2_g)

Imbrication (par-source dans un run global) — chaque tracker collecte
indépendamment ; pas de tracker parent/enfant côté codecarbon (le SDK ne
gère pas ça proprement), donc on désactive les enfants si un parent
externe demande l'exclusivité (`exclusive=True`).
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Détection capabilities (codecarbon, NVML, psutil)
# ---------------------------------------------------------------------------
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker  # type: ignore
    _HAS_CODECARBON = True
except ImportError:
    EmissionsTracker = None  # type: ignore
    OfflineEmissionsTracker = None  # type: ignore
    _HAS_CODECARBON = False

try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except ImportError:
    pynvml = None  # type: ignore
    _HAS_NVML = False

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Hardware / git fingerprint — on l'écrit dans le snapshot pour traçabilité
# ---------------------------------------------------------------------------
def _gpu_info() -> list[dict[str, Any]]:
    """Liste GPUs visibles + nom + VRAM (Mo). Vide si NVML KO."""
    if not _HAS_NVML:
        return []
    info: list[dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            info.append({
                "index": i,
                "name": name,
                "vram_total_mb": mem.total // (1024 * 1024),
            })
        pynvml.nvmlShutdown()
    except Exception:
        return []
    return info


def _cpu_info() -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
    }
    if _HAS_PSUTIL:
        try:
            out["physical_cores"] = psutil.cpu_count(logical=False)
            out["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            pass
    return out


def _git_commit() -> str | None:
    """Hash du commit courant si on est dans un repo Git. Sinon None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()[:12] or None
    except Exception:
        return None
    return None


def hardware_fingerprint() -> dict[str, Any]:
    """Snapshot minimal de la machine — réutilisé dans le registry CSV."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_info(),
        "gpus": _gpu_info(),
        "codecarbon_available": _HAS_CODECARBON,
        "nvml_available": _HAS_NVML,
        "psutil_available": _HAS_PSUTIL,
        "git_commit": _git_commit(),
    }


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
@dataclass
class EnergySnapshot:
    """Photo d'une run : énergie + CO2 + métadata. Sérialisable JSON."""
    run_id: str
    project: str
    task: str
    started_at: float            # timestamp unix (utc)
    duration_s: float            # wall time
    kwh: float                   # énergie totale (CodeCarbon: cpu + gpu + ram)
    kwh_cpu: float
    kwh_gpu: float
    kwh_ram: float
    co2_g: float                 # gCO2eq (région-aware via CodeCarbon)
    country_iso: str | None      # défini par CodeCarbon (geoip si dispo)
    n_samples: int = 0           # rempli par l'appelant — pour KPIs derived
    score: float | None = None   # idem (ex: macro avg)
    extra: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    backend: str = "codecarbon"  # ou "walltime_only" en fallback

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Le tracker lui-même
# ---------------------------------------------------------------------------
class EnergyTracker:
    """Context manager autour de codecarbon.EmissionsTracker."""

    _active_exclusive: "EnergyTracker | None" = None

    def __init__(
        self,
        project: str = "resilient-ai",
        task: str = "default",
        output_dir: str | Path | None = None,
        country_iso_code: str | None = "FRA",
        measure_power_secs: float = 0.5,
        log_level: str = "error",
        exclusive: bool = False,
        n_samples: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.project = project
        self.task = task
        self.output_dir = Path(output_dir) if output_dir else None
        self.country_iso_code = country_iso_code
        self.measure_power_secs = measure_power_secs
        self.log_level = log_level
        self.exclusive = exclusive
        self.n_samples = n_samples
        self.extra = dict(extra or {})

        self.run_id = uuid.uuid4().hex[:12]
        self._cc_tracker: Any = None
        self._t0: float = 0.0
        self._is_noop: bool = False
        self.snapshot: EnergySnapshot | None = None

    def __enter__(self) -> "EnergyTracker":
        if (
            EnergyTracker._active_exclusive is not None
            and not self.exclusive
            and EnergyTracker._active_exclusive is not self
        ):
            self._is_noop = True
            self._t0 = time.time()
            return self

        if self.exclusive:
            EnergyTracker._active_exclusive = self

        self._t0 = time.time()
        if _HAS_CODECARBON:
            try:
                kwargs: dict[str, Any] = {
                    "project_name": f"{self.project}::{self.task}",
                    "measure_power_secs": self.measure_power_secs,
                    "log_level": self.log_level,
                    "save_to_file": False,
                    "tracking_mode": "machine",
                    "allow_multiple_runs": True,
                }
                if self.output_dir is not None:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    kwargs["output_dir"] = str(self.output_dir)
                    kwargs["save_to_file"] = True
                # Si country forcé → OfflineEmissionsTracker (plus de geoip).
                # Sinon EmissionsTracker normal (geoip CodeCarbon).
                if self.country_iso_code:
                    kwargs["country_iso_code"] = self.country_iso_code
                    self._cc_tracker = OfflineEmissionsTracker(**kwargs)
                else:
                    self._cc_tracker = EmissionsTracker(**kwargs)
                self._cc_tracker.start()
            except Exception as e:
                print(f"[energy] codecarbon disabled : {type(e).__name__}: {e}")
                self._cc_tracker = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._t0

        if self._is_noop:
            self.snapshot = self._build_snapshot(
                duration=duration, emissions=None, backend="noop_nested",
            )
            return False

        emissions: float | None = None
        if self._cc_tracker is not None:
            try:
                emissions = self._cc_tracker.stop()
            except Exception as e:
                print(f"[energy] codecarbon stop() failed : {type(e).__name__}: {e}")

        self.snapshot = self._build_snapshot(
            duration=duration,
            emissions=emissions,
            backend="codecarbon" if self._cc_tracker is not None else "walltime_only",
        )

        if self.exclusive and EnergyTracker._active_exclusive is self:
            EnergyTracker._active_exclusive = None

        return False

    def _build_snapshot(self, duration, emissions, backend):
        kwh = kwh_cpu = kwh_gpu = kwh_ram = 0.0
        co2_g = 0.0
        country_iso: str | None = None

        if self._cc_tracker is not None:
            data = getattr(self._cc_tracker, "final_emissions_data", None)
            if data is not None:
                kwh = float(getattr(data, "energy_consumed", 0.0) or 0.0)
                kwh_cpu = float(getattr(data, "cpu_energy", 0.0) or 0.0)
                kwh_gpu = float(getattr(data, "gpu_energy", 0.0) or 0.0)
                kwh_ram = float(getattr(data, "ram_energy", 0.0) or 0.0)
                co2_kg = float(getattr(data, "emissions", emissions or 0.0) or 0.0)
                co2_g = co2_kg * 1000.0
                country_iso = getattr(data, "country_iso_code", None) or None

        return EnergySnapshot(
            run_id=self.run_id,
            project=self.project,
            task=self.task,
            started_at=self._t0,
            duration_s=duration,
            kwh=kwh,
            kwh_cpu=kwh_cpu,
            kwh_gpu=kwh_gpu,
            kwh_ram=kwh_ram,
            co2_g=co2_g,
            country_iso=country_iso,
            n_samples=self.n_samples,
            extra=self.extra,
            hardware=hardware_fingerprint(),
            backend=backend,
        )


# ---------------------------------------------------------------------------
# Helpers : agrégation + KPIs dérivés
# ---------------------------------------------------------------------------
def aggregate_snapshots(
    snapshots: list[EnergySnapshot],
    project: str = "aggregate",
    task: str = "all",
) -> EnergySnapshot:
    """Combine N snapshots en 1 (somme énergie / co2 / samples / time)."""
    if not snapshots:
        return EnergySnapshot(
            run_id=uuid.uuid4().hex[:12], project=project, task=task,
            started_at=time.time(), duration_s=0.0,
            kwh=0.0, kwh_cpu=0.0, kwh_gpu=0.0, kwh_ram=0.0, co2_g=0.0,
            country_iso=None, n_samples=0,
            hardware=hardware_fingerprint(), backend="empty",
        )

    head = snapshots[0]
    backends = {s.backend for s in snapshots}
    return EnergySnapshot(
        run_id=uuid.uuid4().hex[:12],
        project=project,
        task=task,
        started_at=min(s.started_at for s in snapshots),
        duration_s=sum(s.duration_s for s in snapshots),
        kwh=sum(s.kwh for s in snapshots),
        kwh_cpu=sum(s.kwh_cpu for s in snapshots),
        kwh_gpu=sum(s.kwh_gpu for s in snapshots),
        kwh_ram=sum(s.kwh_ram for s in snapshots),
        co2_g=sum(s.co2_g for s in snapshots),
        country_iso=head.country_iso,
        n_samples=sum(s.n_samples for s in snapshots),
        extra={"sub_runs": [s.run_id for s in snapshots], "backends": list(backends)},
        hardware=head.hardware,
        backend=("codecarbon" if backends == {"codecarbon"} else "mixed"),
    )


def derive_efficiency_metrics(snap: EnergySnapshot) -> dict[str, float]:
    """KPIs challenge : tout normalisé par sample ou par score."""
    out: dict[str, float] = {}
    joules = snap.kwh * 3_600_000.0
    n = max(snap.n_samples, 1)
    out["joules_per_sample"] = joules / n
    out["samples_per_joule"] = n / joules if joules > 0 else 0.0
    out["co2_mg_per_sample"] = (snap.co2_g * 1000.0) / n
    out["wh_per_sample"] = (snap.kwh * 1000.0) / n
    if snap.score is not None and snap.score > 0:
        out["score_per_kwh"] = snap.score / snap.kwh if snap.kwh > 0 else 0.0
        out["kwh_per_score_unit"] = snap.kwh / snap.score
    return out


def maybe_tracker(enabled: bool, **kwargs: Any):
    """EnergyTracker si enabled=True sinon nullcontext()."""
    if not enabled:
        return nullcontext()
    return EnergyTracker(**kwargs)


__all__ = [
    "EnergyTracker",
    "EnergySnapshot",
    "aggregate_snapshots",
    "derive_efficiency_metrics",
    "hardware_fingerprint",
    "maybe_tracker",
]
