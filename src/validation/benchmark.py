"""
Benchmark unifié — évalue un Student (ou tout modèle) sur les sources du
dataset unifié, avec tracking énergie CodeCarbon par-source.

Principe : val/test sont **par-source** (pas mélangés). Le benchmark boucle
dataset par dataset, dispatche vers la bonne métrique, puis agrège :
- micro-average (pondéré par taille)
- macro-average (poids égal par dataset)

Décision Mathis 2026-04-30 :
  - Le bench tourne par défaut sur **10% du dataset unifié** (subsample
    déterministe par-source, seed=42). Itération rapide, comparaisons
    reproductibles entre runs.
  - On retire MMStar / MME / POPE du flow de benchmark — ce sont des
    benchmarks OOD externes qui restent dispo dans `eval_ood.py` pour
    usage manuel ponctuel mais ne sont PLUS appelés par `eval_loop.py`.
  - Tracking énergie ajouté à chaque source via `EnergyTracker`.

Usage (Runpod) :

    python run_benchmark.py \
        --model_path runs/student-quantized-int4 \
        --dataset_hub Mathos34400/resilient-ai-unified \
        --split test \
        --subsample_pct 10 \
        --out results/qat-int4.json
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Iterable

from src.validation.metrics import (
    score_sample,
    cider_corpus,
    normalize_answer,
)
from src.tracking.energy_tracker import (
    EnergyTracker,
    EnergySnapshot,
    aggregate_snapshots,
    derive_efficiency_metrics,
)


# Par défaut : 10% du dataset unifié (test split), seed fixe = comparable
# entre runs successives sur le même split. Override via CLI / argument.
DEFAULT_SUBSAMPLE_PCT: float = 10.0
DEFAULT_SUBSAMPLE_SEED: int = 42


# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------
GenerateFn = Callable[..., str]


@dataclass
class PerSourceResult:
    source: str
    n_samples: int
    n_scored: int
    score: float
    metric: str
    extra: dict[str, Any]
    energy: dict[str, Any] | None = None  # snap.to_dict() si tracking actif


@dataclass
class BenchmarkResult:
    model_id: str
    split: str
    timestamp: float
    per_source: list[PerSourceResult]
    micro_avg: float
    macro_avg: float
    wall_time_s: float
    commit: str | None = None
    # Énergie agrégée (somme des per-source snapshots)
    energy_total: dict[str, Any] | None = None
    efficiency_kpi: dict[str, float] | None = None
    subsample_pct: float = 100.0
    subsample_seed: int | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                **asdict(self),
                "per_source": [asdict(p) for p in self.per_source],
            },
            indent=2,
        )


# ----------------------------------------------------------------------------
# Subsample déterministe par-source
# ----------------------------------------------------------------------------
def _stable_hash(value: Any) -> int:
    """Hash stable inter-runs (le hash builtin Python varie avec PYTHONHASHSEED)."""
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8], 16)


def subsample_per_source(
    samples_by_src: dict[str, list[dict]],
    pct: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Garde `pct`% des samples de chaque source.

    Stratégie : on shuffle déterministiquement (seed * hash(source)) puis
    on garde les N premiers — ça garantit :
      1. reproductibilité inter-runs (même split → mêmes samples)
      2. indépendance entre sources (deux sources tirent des indices
         différents, sinon on aurait toujours le sample id=0 de chaque)
      3. au moins 1 sample par source si la source est non-vide

    pct : entre 0 (exclu) et 100. 100 → no-op.
    """
    if pct >= 100.0:
        return samples_by_src
    if pct <= 0.0:
        raise ValueError("subsample pct doit être > 0")

    out: dict[str, list[dict]] = {}
    for src, lst in samples_by_src.items():
        if not lst:
            out[src] = lst
            continue
        rng = random.Random(seed ^ _stable_hash(src))
        idx = list(range(len(lst)))
        rng.shuffle(idx)
        keep = max(1, int(round(len(lst) * pct / 100.0)))
        out[src] = [lst[i] for i in idx[:keep]]
    return out


# ----------------------------------------------------------------------------
# Boucle d'évaluation
# ----------------------------------------------------------------------------
def evaluate_source(
    samples: Iterable[dict],
    generate_fn: GenerateFn,
    source: str,
    metric_hint: str,
    progress: bool = True,
    batch_size: int = 1,
    track_energy: bool = True,
    project_label: str = "bench",
    country_iso: str = "FRA",
) -> tuple[PerSourceResult, EnergySnapshot | None]:
    """Évalue un modèle sur un jeu de samples d'un même source.

    Retourne (PerSourceResult, EnergySnapshot|None). Le snapshot est None
    si `track_energy=False` ou si codecarbon a échoué.
    """
    samples = list(samples)
    n = len(samples)
    if n == 0:
        return PerSourceResult(source, 0, 0, 0.0, metric_hint, {}, None), None

    preds: list[str] = []
    gts: list[list[str]] = []
    per_sample_scores: list[float] = []

    iterator = samples
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc=source, leave=False)
        except ImportError:
            pass

    snap: EnergySnapshot | None = None
    tracker = EnergyTracker(
        project=project_label,
        task=source,
        n_samples=n,
        country_iso_code=country_iso,
    ) if track_energy else None

    if tracker is not None:
        tracker.__enter__()
    try:
        for sample in iterator:
            image = sample.get("image")
            prompt = sample.get("question")
            pred = generate_fn(image=image, prompt=prompt, sample=sample)
            preds.append(pred)
            gts.append(sample.get("answers") or [])

            if metric_hint != "cider":
                s = score_sample(metric_hint, pred, sample)
                if s == s:  # filtre NaN
                    per_sample_scores.append(s)
    finally:
        if tracker is not None:
            tracker.__exit__(None, None, None)
            snap = tracker.snapshot
            if snap is not None:
                snap.n_samples = n

    if metric_hint == "cider":
        score = cider_corpus(preds, gts)
        n_scored = n
        extra = {"n_preds": len(preds)}
    else:
        if not per_sample_scores:
            return PerSourceResult(
                source, n, 0, 0.0, metric_hint, {},
                snap.to_dict() if snap else None,
            ), snap
        score = sum(per_sample_scores) / len(per_sample_scores)
        n_scored = len(per_sample_scores)
        extra = {}

    if snap is not None:
        snap.score = score

    return (
        PerSourceResult(
            source=source,
            n_samples=n,
            n_scored=n_scored,
            score=score,
            metric=metric_hint,
            extra=extra,
            energy=snap.to_dict() if snap else None,
        ),
        snap,
    )


def evaluate_model(
    dataset,                           # datasets.Dataset (split val ou test)
    generate_fn: GenerateFn,
    model_id: str,
    split: str,
    limit_per_source: int | None = None,
    sources: list[str] | None = None,
    subsample_pct: float = DEFAULT_SUBSAMPLE_PCT,
    subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
    track_energy: bool = True,
    project_label: str = "bench",
    country_iso: str = "FRA",
) -> tuple[BenchmarkResult, list[EnergySnapshot]]:
    """Exécute le benchmark sur les sources présentes dans `dataset`.

    Args:
        subsample_pct : 10.0 par défaut (cf. DEFAULT_SUBSAMPLE_PCT) — on
                        tire 10% par-source de manière déterministe.
                        Mettre 100 pour passer le full split.
        track_energy  : True = wrap chaque source dans un EnergyTracker
                        + agrège un snapshot total en fin de run.
        limit_per_source : appliqué APRÈS le subsample (cap dur, utile
                           smoke test rapide).

    Returns:
        (BenchmarkResult, list[EnergySnapshot]) — la liste contient un
        snapshot par-source pour pouvoir dumper le détail dans le registry.
    """
    t0 = time.time()

    # 1. Regroupe par source
    by_src: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        src = row["source"]
        if sources and src not in sources:
            continue
        by_src[src].append(row)

    # 2. Subsample déterministe par-source
    if subsample_pct < 100.0:
        before = sum(len(v) for v in by_src.values())
        by_src = subsample_per_source(by_src, subsample_pct, subsample_seed)
        after = sum(len(v) for v in by_src.values())
        print(f"  [subsample] {subsample_pct}% (seed={subsample_seed}) : "
              f"{before} → {after} samples")

    # 3. Cap éventuel (post-subsample)
    if limit_per_source:
        by_src = {k: v[:limit_per_source] for k, v in by_src.items()}

    # 4. Boucle par-source
    results: list[PerSourceResult] = []
    snapshots: list[EnergySnapshot] = []
    for src, samples in sorted(by_src.items()):
        hint = samples[0]["metric_hint"]
        print(f"  → {src}: {len(samples)} samples (metric={hint})")
        r, snap = evaluate_source(
            samples, generate_fn, src, hint,
            track_energy=track_energy,
            project_label=project_label,
            country_iso=country_iso,
        )
        results.append(r)
        if snap is not None:
            snapshots.append(snap)
        e_str = ""
        if snap is not None and snap.kwh > 0:
            e_str = f"  [{snap.kwh*1000:.3f} Wh / {snap.duration_s:.1f}s]"
        print(f"    score = {r.score:.4f}  (scored {r.n_scored}/{r.n_samples}){e_str}")

    # 5. Agrégats scores
    valid = [r for r in results if r.n_scored > 0 and r.score == r.score]
    if valid:
        total = sum(r.n_scored for r in valid)
        micro = sum(r.score * r.n_scored for r in valid) / total
        macro = sum(r.score for r in valid) / len(valid)
    else:
        micro = macro = 0.0

    # 6. Agrégat énergie
    energy_total_dict = None
    efficiency_kpi = None
    if snapshots:
        agg = aggregate_snapshots(snapshots, project=project_label, task="all_sources")
        agg.score = macro
        agg.n_samples = sum(r.n_samples for r in results)
        energy_total_dict = agg.to_dict()
        efficiency_kpi = derive_efficiency_metrics(agg)

    return (
        BenchmarkResult(
            model_id=model_id,
            split=split,
            timestamp=t0,
            per_source=results,
            micro_avg=micro,
            macro_avg=macro,
            wall_time_s=time.time() - t0,
            energy_total=energy_total_dict,
            efficiency_kpi=efficiency_kpi,
            subsample_pct=subsample_pct,
            subsample_seed=subsample_seed,
        ),
        snapshots,
    )


# ----------------------------------------------------------------------------
# Adaptateurs d'inférence
# ----------------------------------------------------------------------------

def make_gemma_generate_fn(model, processor, max_new_tokens: int = 32, device: str = "cuda"):
    """Retourne une GenerateFn alignée sur les métriques du benchmark.

    Trois adaptations critiques par rapport à la naive version :
      - Multiple choice : on injecte les choices avec lettres A/B/C/D et on
        force "ONLY the letter" sinon Gemma répond par le contenu textuel
        ("a dog") au lieu d'une lettre.
      - VQA : Gemma 4-it est très verbeux ("The answer is **plaid** or
        checked..."). On instruit "as few words as possible, no markdown,
        no explanation" + on post-traite pour stripper le markdown et
        couper à la 1ère ligne.
      - Captioning : on borne à une phrase courte pour que CIDEr/BLEU ne
        soient pas écrasés par les paragraphes descriptifs.

    Compatible Gemma 3n / 4-it (transformers ≥ 4.44).
    """
    import torch

    def _gen(image, prompt, sample=None, **kwargs) -> str:
        s = sample or {}
        task = s.get("task", "vqa")
        choices = s.get("choices") or []

        if task == "multiple_choice" and choices:
            labeled = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
            text = (
                f"{(prompt or '').strip()}\n\n{labeled}\n\n"
                "Answer with ONLY the letter (A, B, C, ...) of the correct choice."
            )
        elif task == "captioning":
            text = "Describe this image in one short sentence. No explanation, no markdown."
        else:  # vqa
            text = (
                f"{(prompt or '').strip()}\n\n"
                "Answer in as few words as possible. No explanation, no markdown."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[:, inputs["input_ids"].shape[-1]:]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return _post_process_pred(raw, task)

    return _gen


def _post_process_pred(text: str, task: str) -> str:
    """Nettoie la réponse Gemma pour matcher les ground truth normalisés.

    - strippe le markdown bold/italic (**...**, *...*)
    - garde uniquement la 1ère ligne (Gemma adore enchaîner avec des
      bullet points "Here's why: ...")
    - pour les MC, extrait la première lettre A-E si présente
    """
    import re as _re
    text = _re.sub(r"\*+", "", text).strip()
    text = text.split("\n", 1)[0].strip()
    if task == "multiple_choice":
        m = _re.search(r"\b([A-E])\b", text)
        if m:
            return m.group(1)
    return text


# ----------------------------------------------------------------------------
# Dry-run stub (sans modèle) — utile pour sanity-check les métriques
# ----------------------------------------------------------------------------

def constant_generate_fn(answer: str = "unknown") -> GenerateFn:
    def _gen(image=None, prompt=None, sample=None, **kwargs) -> str:
        return answer
    return _gen


def oracle_generate_fn() -> GenerateFn:
    def _gen(image=None, prompt=None, sample=None, **kwargs) -> str:
        if sample is None:
            return ""
        if sample.get("task") == "multiple_choice" and sample.get("correct_choice_idx") is not None:
            return chr(65 + int(sample["correct_choice_idx"]))
        ans = sample.get("answers") or []
        return ans[0] if ans else ""
    return _gen
