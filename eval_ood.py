"""
Évaluation OOD (Out-Of-Distribution) — Layer 2 "gold signal" pour le challenge.

Benchmarks externes que le modèle n'a **jamais** vus :
  - ni dans train mix
  - ni dans calibration PTQ
  - ni dans val/test in-distribution

Ce sont les seuls indicateurs fiables de la performance sur les tests
challenge (que le jury fait probablement tourner sur des benchmarks
publics différents de ce qu'on utilise en train).

Règle d'or INVIOLABLE : aucun de ces datasets ne doit apparaître dans
config.py / ALL_DATASETS. Si tu en as besoin pour autre chose (ex: probe
de représentations), garde-le strictement hors du pipeline train.

Benchmarks implémentés :
  - MMStar   (Lin-Chen/MMStar, 1500 samples, 6 catégories, MC)
  - MME      (lmms-lab/MME, 2374 samples, 14 sous-tâches Yes/No)
  - POPE     (lmms-lab/POPE, 9000 samples, hallucination Yes/No F1)

Utilisation :
    from eval_ood import evaluate_ood_all
    results = evaluate_ood_all(
        generate_fn,
        benches=["mmstar", "mme", "pope"],
        limit_per_bench=None,  # None = tout ; int = cap (smoke test)
    )
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Iterable

# Ces IDs HF peuvent évoluer ; on maintient ici les canoniques mi-2026.
OOD_HF_IDS = {
    "mmstar": ("Lin-Chen/MMStar", "val"),
    "mme": ("lmms-lab/MME", "test"),
    "pope": ("lmms-lab/POPE", "test"),
}

# Alternatives si le nom primaire est down (plan B, itéré à l'exécution)
OOD_HF_FALLBACKS = {
    "mmstar": [("Lin-Chen/MMStar", "val")],
    "mme": [
        ("lmms-lab/MME", "test"),
        ("lmms-lab/MME", "cognition_perception_test"),
    ],
    "pope": [
        ("lmms-lab/POPE", "test"),
        ("lmms-lab/POPE", "popular"),
    ],
}


GenerateFn = Callable[..., str]


# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    name: str
    n_samples: int
    n_scored: int
    score: float                 # indicateur principal (accuracy, F1, ou score/2800)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OodReport:
    timestamp: float
    model_id: str
    per_bench: list[BenchResult]
    wall_time_s: float

    @property
    def aggregate_score(self) -> float:
        """Moyenne simple des scores normalisés [0, 1] sur les benches."""
        valid = [b for b in self.per_bench if b.n_scored > 0]
        return sum(b.score for b in valid) / len(valid) if valid else 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "model_id": self.model_id,
                "aggregate_score": self.aggregate_score,
                "per_bench": [asdict(b) for b in self.per_bench],
                "wall_time_s": self.wall_time_s,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------
def _load_ood(bench: str):
    """Charge le dataset OOD avec fallback si l'ID primaire échoue."""
    from datasets import load_dataset

    for hf_id, split in OOD_HF_FALLBACKS.get(bench, [OOD_HF_IDS[bench]]):
        try:
            return load_dataset(hf_id, split=split), hf_id, split
        except Exception as e:
            print(f"  [ood] {hf_id}:{split} → échec ({type(e).__name__}), fallback…")
    raise RuntimeError(f"Impossible de charger le benchmark OOD '{bench}'")


def _maybe_tqdm(it: Iterable, desc: str, progress: bool = True) -> Iterable:
    if not progress:
        return it
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, leave=False)
    except ImportError:
        return it


def _parse_yn(pred: str) -> str | None:
    """Extrait 'yes' ou 'no' d'une réponse VLM (robuste au verbose)."""
    p = (pred or "").lower().strip()
    if not p:
        return None
    m = re.match(r"\b(yes|yeah|yep|correct|true)\b", p)
    if m:
        return "yes"
    m = re.match(r"\b(no|nope|incorrect|false)\b", p)
    if m:
        return "no"
    # Cherche dans les 5 premiers tokens
    toks = re.findall(r"\b\w+\b", p)[:5]
    for t in toks:
        if t in ("yes", "no"):
            return t
    return None


def _parse_letter(pred: str, letters: str = "ABCD") -> str | None:
    """Extrait une lettre MC (A/B/C/D) d'une réponse VLM."""
    p = (pred or "").strip().upper()
    if not p:
        return None
    # Cas trivial : premier char est la lettre
    if p[0] in letters and (len(p) == 1 or not p[1].isalpha()):
        return p[0]
    # Patterns : "(A)", "A.", "A)", "answer is A", "A:"
    patterns = [
        rf"\(([{letters}])\)",
        rf"ANSWER\s+IS\s+([{letters}])\b",
        rf"\b([{letters}])\s*[\.\)\:]",
        rf"\b([{letters}])\b",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# MMStar — Lin-Chen/MMStar
# ---------------------------------------------------------------------------
def evaluate_ood_mmstar(
    generate_fn: GenerateFn,
    limit: int | None = None,
    progress: bool = True,
) -> BenchResult:
    """
    MMStar (Chen et al., 2024) — 1500 samples, 6 catégories, MC vision-required.
    Chaque sample : image, question, options (dans la question ou séparées),
    answer (lettre A-D), category.

    Métrique : accuracy globale + per-category.
    """
    print("[ood] MMStar — loading…")
    ds, hf_id, split = _load_ood("mmstar")
    samples = list(ds)
    if limit:
        samples = samples[:limit]
    n = len(samples)
    if n == 0:
        return BenchResult("mmstar", 0, 0, 0.0)

    scores: list[int] = []
    per_cat: dict[str, list[int]] = defaultdict(list)

    for s in _maybe_tqdm(samples, "mmstar", progress):
        img = s.get("image")
        # MMStar stocke souvent la question AVEC les options inline.
        question = s.get("question") or ""
        prompt = (
            question + "\n"
            "Answer with only the letter of the correct option (A, B, C, or D)."
        )
        pred = generate_fn(image=img, prompt=prompt, sample=s)
        pred_letter = _parse_letter(pred, "ABCD")
        gold = (s.get("answer") or "").strip().upper()
        if gold and len(gold) > 0:
            gold = gold[0]
        correct = int(pred_letter == gold) if (pred_letter and gold) else 0
        scores.append(correct)
        cat = s.get("category") or s.get("meta_category") or "all"
        per_cat[cat].append(correct)

    acc = sum(scores) / len(scores) if scores else 0.0
    per_cat_acc = {
        c: (sum(v) / len(v) if v else 0.0) for c, v in per_cat.items()
    }
    return BenchResult(
        name="mmstar",
        n_samples=n,
        n_scored=len(scores),
        score=float(acc),
        extra={"per_category": per_cat_acc, "hf_id": hf_id, "split": split},
    )


# ---------------------------------------------------------------------------
# MME — lmms-lab/MME
# ---------------------------------------------------------------------------
# Les 14 sous-tâches sont réparties perception (10) vs cognition (4).
_MME_PERCEPTION = {
    "existence", "count", "position", "color", "poster", "celebrity",
    "scene", "landmark", "artwork", "OCR",
}
_MME_COGNITION = {
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
}


def evaluate_ood_mme(
    generate_fn: GenerateFn,
    limit: int | None = None,
    progress: bool = True,
) -> BenchResult:
    """
    MME (Fu et al., 2023) — 14 sous-tâches Yes/No.
    Score officiel = somme des accuracies par sous-tâche × 200 (max 2800).

    Simplifications ici :
      - accuracy par sous-tâche
      - "perception_score" = somme des 10 perception × 200  (max 2000)
      - "cognition_score" = somme des 4 cognition × 200     (max 800)
      - `score` principal = (perception_score + cognition_score) / 2800

    Note : MME utilise Yes/No pair accuracy (plus stricte) dans le scoring
    officiel (les 2 questions du pair doivent être correctes). On
    implémente la version simple per-sample ici pour vitesse ; la version
    pair-accuracy est une amélioration facile si besoin.
    """
    print("[ood] MME — loading…")
    ds, hf_id, split = _load_ood("mme")
    samples = list(ds)
    if limit:
        samples = samples[:limit]
    n = len(samples)
    if n == 0:
        return BenchResult("mme", 0, 0, 0.0)

    per_task: dict[str, list[int]] = defaultdict(list)

    for s in _maybe_tqdm(samples, "mme", progress):
        img = s.get("image")
        question = s.get("question") or ""
        task = s.get("category") or s.get("task") or s.get("question_type") or "unknown"
        prompt = question + "\nAnswer with only 'yes' or 'no'."
        pred = generate_fn(image=img, prompt=prompt, sample=s)
        pred_yn = _parse_yn(pred)
        gold_yn = (s.get("answer") or "").lower().strip()
        if gold_yn.startswith("y"):
            gold_yn = "yes"
        elif gold_yn.startswith("n"):
            gold_yn = "no"
        correct = int(pred_yn == gold_yn) if (pred_yn and gold_yn) else 0
        per_task[task].append(correct)

    per_task_acc = {t: sum(v) / len(v) for t, v in per_task.items() if v}
    perception_score = sum(acc * 200 for t, acc in per_task_acc.items()
                           if t in _MME_PERCEPTION)
    cognition_score = sum(acc * 200 for t, acc in per_task_acc.items()
                          if t in _MME_COGNITION)
    mme_total = perception_score + cognition_score
    normalized = mme_total / 2800.0

    return BenchResult(
        name="mme",
        n_samples=n,
        n_scored=sum(len(v) for v in per_task.values()),
        score=float(normalized),
        extra={
            "perception_score": float(perception_score),
            "cognition_score": float(cognition_score),
            "mme_total": float(mme_total),
            "per_task_accuracy": per_task_acc,
            "hf_id": hf_id,
            "split": split,
        },
    )


# ---------------------------------------------------------------------------
# POPE — lmms-lab/POPE
# ---------------------------------------------------------------------------
def evaluate_ood_pope(
    generate_fn: GenerateFn,
    limit: int | None = None,
    progress: bool = True,
) -> BenchResult:
    """
    POPE (Li et al., 2023) — hallucination d'objets, format Yes/No.
    3 sous-splits : random / popular / adversarial (sur MSCOCO).

    Métriques : accuracy, precision, recall, F1 (positive class = 'yes').
    Score principal = F1 moyen sur les sous-splits disponibles.
    """
    print("[ood] POPE — loading…")
    ds, hf_id, split = _load_ood("pope")
    samples = list(ds)
    if limit:
        samples = samples[:limit]
    n = len(samples)
    if n == 0:
        return BenchResult("pope", 0, 0, 0.0)

    # Regroupe par sous-split si dispo
    by_sub: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unparsed": 0, "n": 0}
    )

    for s in _maybe_tqdm(samples, "pope", progress):
        img = s.get("image")
        question = s.get("question") or ""
        prompt = question + "\nAnswer with only 'yes' or 'no'."
        pred = generate_fn(image=img, prompt=prompt, sample=s)
        pred_yn = _parse_yn(pred)
        gold_yn = (s.get("answer") or "").lower().strip()
        if gold_yn.startswith("y"):
            gold_yn = "yes"
        elif gold_yn.startswith("n"):
            gold_yn = "no"
        sub = s.get("category") or s.get("subset") or s.get("type") or "all"
        cell = by_sub[sub]
        cell["n"] += 1
        if pred_yn is None:
            cell["unparsed"] += 1
            pred_yn = "no"  # pénalise non-réponses
        if gold_yn == "yes" and pred_yn == "yes":
            cell["tp"] += 1
        elif gold_yn == "no" and pred_yn == "yes":
            cell["fp"] += 1
        elif gold_yn == "yes" and pred_yn == "no":
            cell["fn"] += 1
        else:
            cell["tn"] += 1

    sub_metrics: dict[str, dict[str, float]] = {}
    for sub, c in by_sub.items():
        total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        acc = (c["tp"] + c["tn"]) / total if total else 0.0
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sub_metrics[sub] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "unparsed_frac": c["unparsed"] / c["n"] if c["n"] else 0.0,
            "n": c["n"],
        }

    # Score principal = F1 moyen sur sous-splits
    f1_mean = (
        sum(m["f1"] for m in sub_metrics.values()) / len(sub_metrics)
        if sub_metrics else 0.0
    )

    return BenchResult(
        name="pope",
        n_samples=n,
        n_scored=sum(c["n"] for c in by_sub.values()),
        score=float(f1_mean),
        extra={
            "per_subset": sub_metrics,
            "hf_id": hf_id,
            "split": split,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------
def evaluate_ood_all(
    generate_fn: GenerateFn,
    model_id: str = "unknown",
    benches: list[str] | None = None,
    limit_per_bench: int | None = None,
    progress: bool = True,
) -> OodReport:
    """
    Évalue le modèle sur tous les benchmarks OOD sélectionnés.

    Args:
        generate_fn      : cf. benchmark.GenerateFn
        model_id         : label pour le report
        benches          : subset parmi {mmstar, mme, pope}. None = tout.
        limit_per_bench  : cap samples (smoke test rapide ; None = full).

    Returns:
        OodReport avec per_bench et aggregate_score.
    """
    benches = benches or list(OOD_HF_IDS.keys())
    t0 = time.time()
    results: list[BenchResult] = []

    dispatch = {
        "mmstar": evaluate_ood_mmstar,
        "mme": evaluate_ood_mme,
        "pope": evaluate_ood_pope,
    }

    for b in benches:
        if b not in dispatch:
            print(f"[ood] ⚠ bench inconnu : {b} (ignoré)")
            continue
        print(f"\n[ood] === {b.upper()} ===")
        try:
            r = dispatch[b](generate_fn, limit=limit_per_bench, progress=progress)
            results.append(r)
            print(f"[ood] {b}: score = {r.score:.4f}  "
                  f"(n_scored={r.n_scored}/{r.n_samples})")
        except Exception as e:
            print(f"[ood] {b} a échoué : {type(e).__name__}: {e}")
            results.append(BenchResult(b, 0, 0, 0.0, extra={"error": str(e)}))

    return OodReport(
        timestamp=t0,
        model_id=model_id,
        per_bench=results,
        wall_time_s=time.time() - t0,
    )
