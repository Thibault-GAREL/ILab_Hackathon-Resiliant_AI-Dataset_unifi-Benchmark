"""
Stratégies de mix + découpage train/val/test.

4 stratégies d'échantillonnage supportées (décision équipe non tranchée au
18/04/2026) :

1. uniform       : w_i ∝ n_i                (pondération par taille — biais gros)
2. sqrt          : w_i ∝ √n_i               (standard multi-task VLM)
3. task_balanced : w_i = 1/N                (poids égal par dataset)
4. temperature   : w_i ∝ n_i^(1/T)          (tunable, T=1 ≡ uniform, T→∞ ≡ task_bal.)

Convention splits :
- si le dataset a des val/test officiels (DatasetSpec.splits), on les utilise
  pour val/test — c'est la bonne pratique pour que les résultats soient
  comparables aux leaderboards publics ;
- si pas de split officiel (ex : OK-VQA train seul, RealWorldQA test-only),
  on découpe 80/10/10 en interne avec un seed fixe.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator

from config import ALL_DATASETS, MixConfig, SplitConfig, MixStrategy
from loaders import load_source
from schema import UnifiedSample


# ----------------------------------------------------------------------------
# Poids
# ----------------------------------------------------------------------------

def compute_weights(sizes: dict[str, int], cfg: MixConfig) -> dict[str, float]:
    """Retourne une distribution normalisée sur les datasets présents.

    `sizes` : dict nom_dataset → nb de samples disponibles dans le split train.
    Retour : dict de même clé, somme = 1.0.
    """
    if not sizes:
        raise ValueError("Aucun dataset à mélanger.")

    strat: MixStrategy = cfg.strategy
    if strat == "uniform":
        raw = {k: float(v) for k, v in sizes.items()}
    elif strat == "sqrt":
        raw = {k: math.sqrt(v) for k, v in sizes.items()}
    elif strat == "task_balanced":
        raw = {k: 1.0 for k in sizes}
    elif strat == "temperature":
        T = max(cfg.temperature, 1e-6)
        raw = {k: v ** (1.0 / T) for k, v in sizes.items()}
    else:
        raise ValueError(f"Stratégie inconnue : {strat}")

    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ----------------------------------------------------------------------------
# Sampler avec remplacement (évite de materialiser 1M+ VQAv2 en RAM)
# ----------------------------------------------------------------------------

class WeightedMixSampler:
    """Sampler qui tire des samples selon les poids, sur pools indexés.

    Usage :
        pools = {name: [UnifiedSample, ...], ...}   # train sets matérialisés
        sampler = WeightedMixSampler(pools, weights, seed=42)
        for sample in sampler.take(n_steps):
            ...

    Implémentation : on utilise random.choices (avec remise) qui est O(1) par
    tirage après setup. Pour contrôler l'empreinte mémoire sur VQAv2, le
    build_dataset.py peut (optionnellement) cap chaque pool à
    `MixConfig.max_per_dataset` avant construction du sampler.
    """

    def __init__(
        self,
        pools: dict[str, list[UnifiedSample]],
        weights: dict[str, float],
        seed: int = 42,
    ) -> None:
        self.pools = {k: v for k, v in pools.items() if v}
        if not self.pools:
            raise ValueError("Tous les pools sont vides.")
        self.weights = {k: weights[k] for k in self.pools}
        # renormalisation si des datasets ont été filtrés
        s = sum(self.weights.values())
        self.weights = {k: v / s for k, v in self.weights.items()}
        self._names = list(self.weights.keys())
        self._ws = [self.weights[k] for k in self._names]
        self._rng = random.Random(seed)

    def take(self, n: int) -> Iterator[UnifiedSample]:
        for _ in range(n):
            src = self._rng.choices(self._names, weights=self._ws, k=1)[0]
            pool = self.pools[src]
            idx = self._rng.randrange(len(pool))
            yield pool[idx]

    def effective_sizes(self, n_total: int) -> dict[str, int]:
        """Répartition attendue pour n_total samples tirés."""
        return {k: int(round(n_total * w)) for k, w in self.weights.items()}


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def _split_list(
    items: list[UnifiedSample],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[UnifiedSample], list[UnifiedSample], list[UnifiedSample]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "Les ratios doivent sommer à 1."
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_source_splits(
    dataset_names: list[str],
    split_cfg: SplitConfig,
    max_per_split: int | None = None,
    eval_cap: int | None = None,
) -> dict[str, dict[str, list[UnifiedSample]]]:
    """Pour chaque dataset, retourne un dict {'train','val','test'} de listes.

    Respecte les splits officiels quand ils existent (split_cfg.use_official_splits=True),
    sinon fallback 80/10/10 à partir du split train.

    `max_per_split` : cap streaming sur le split *train* (évite DL complet de
    VQAv2 etc.). Laisser None pour Runpod si disque suffisant.
    `eval_cap` : cap sur val/test (par défaut : max_per_split // 10 si set,
    sinon None — permet de ne pas exploser la RAM sur les grosses val).
    """
    if eval_cap is None and max_per_split is not None:
        eval_cap = max(200, max_per_split // 5)

    out: dict[str, dict[str, list[UnifiedSample]]] = {}
    for name in dataset_names:
        print(f"  [load] {name} ...", flush=True)
        spec = ALL_DATASETS[name]
        splits: dict[str, list[UnifiedSample]] = {"train": [], "val": [], "test": []}

        official_train = spec.splits.get("train")
        official_val = spec.splits.get("val")
        official_test = spec.splits.get("test")

        def _safe_load(which: str, budget: int | None) -> list[UnifiedSample]:
            print(f"    - {which} (budget={budget}) ...", flush=True)
            try:
                out_list = list(load_source(name, which, max_samples=budget))
                print(f"    - {which}: {len(out_list)} samples", flush=True)
                return out_list
            except Exception as e:
                print(f"    ! {name}.{which}: skip ({type(e).__name__}: {str(e)[:160]})", flush=True)
                return []

        if split_cfg.use_official_splits and official_train and (official_val or official_test):
            # Cas nominal : splits natifs exploitables
            splits["train"] = _safe_load("train", max_per_split) if official_train else []
            splits["val"] = _safe_load("val", eval_cap) if official_val else []
            splits["test"] = _safe_load("test", eval_cap) if official_test else []
            # Si l'un manque, on en reconstruit depuis train (éviter train/val/test vide)
            if not splits["val"] or not splits["test"]:
                tr, va, te = _split_list(splits["train"], split_cfg.ratios, split_cfg.seed)
                splits["train"] = tr
                splits["val"] = splits["val"] or va
                splits["test"] = splits["test"] or te
        elif official_train:
            # Fallback : split 80/10/10 sur le train natif
            items = _safe_load("train", max_per_split)
            tr, va, te = _split_list(items, split_cfg.ratios, split_cfg.seed)
            splits["train"], splits["val"], splits["test"] = tr, va, te
        elif official_test:
            # Eval-only (ex: RealWorldQA)
            splits["test"] = _safe_load("test", eval_cap)
        else:
            print(f"  ! {name}: aucun split officiel (skip)")
            continue

        out[name] = splits
    return out


# ----------------------------------------------------------------------------
# Composition finale
# ----------------------------------------------------------------------------

def compose_train(
    source_splits: dict[str, dict[str, list[UnifiedSample]]],
    mix_cfg: MixConfig,
    n_samples: int,
) -> list[UnifiedSample]:
    """Produit la liste finale du train set unifié, échantillonnée."""
    pools: dict[str, list[UnifiedSample]] = {}
    for name, sp in source_splits.items():
        pool = sp.get("train") or []
        if mix_cfg.max_per_dataset is not None:
            pool = pool[: mix_cfg.max_per_dataset]
        if pool:
            pools[name] = pool

    sizes = {k: len(v) for k, v in pools.items()}
    weights = compute_weights(sizes, mix_cfg)
    sampler = WeightedMixSampler(pools, weights, seed=mix_cfg.seed)
    mixed = list(sampler.take(n_samples))
    return mixed


def compose_eval(
    source_splits: dict[str, dict[str, list[UnifiedSample]]],
    which: str,  # 'val' | 'test'
) -> dict[str, list[UnifiedSample]]:
    """Val/Test : on NE mélange PAS. Chaque dataset reste séparé pour éval par-source."""
    assert which in ("val", "test")
    return {name: sp.get(which, []) for name, sp in source_splits.items() if sp.get(which)}


def compose_calibration(
    source_splits: dict[str, dict[str, list[UnifiedSample]]],
    n_total: int = 2048,
    seed: int = 42,
    remove_from_pools: bool = True,
) -> list[UnifiedSample]:
    """Construit un split de calibration stratifié pour quantization PTQ.

    Stratégie : tirage sans remise équitable par source (n_total // n_sources
    par source), puis shuffle global pour casser les blocs mono-source qui
    biaiseraient les statistiques d'activation lors du calibrage AWQ/GPTQ.

    Pourquoi séparer de train mix :
    - la quantization PTQ (AWQ, GPTQ, SmoothQuant) calibre ses échelles sur
      quelques centaines à quelques milliers de samples ; les papiers AWQ et
      GPTQ convergent sur 128-1024 comme fenêtre optimale, au-delà le gain est
      marginal. 2048 = marge confortable + couvre les 12 sources en stratifié ;
    - ces samples doivent être **distincts** du train set pour éviter toute
      fuite (un sample calibré puis rencontré en fine-tuning ferait surfiter
      les échelles sur du data-already-seen par le student) ;
    - l'équipe quantization (Achta, Djebril) peut pointer directement sur le
      split `calibration` du Hub sans rejouer l'échantillonnage.

    Args :
        source_splits : sortie de `build_source_splits` (on pioche dans train).
        n_total : taille visée de la calibration (défaut 2048).
        seed : graine pour reproductibilité (différente de celle du mix pour
               éviter corrélation indésirable avec le train sampling).
        remove_from_pools : si True (défaut), retire physiquement les samples
               piochés des listes `source_splits[*]["train"]` pour garantir
               la disjonction calibration ⟂ train. DOIT être appelée AVANT
               `compose_train`.

    Returns :
        Liste mélangée d'UnifiedSample, tirée SANS remise pool-par-pool.
        Taille réelle peut être < n_total si les pools sont trop petits.
    """
    # Ne retenir que les sources avec un train non-vide
    train_sources = {
        name: sp["train"] for name, sp in source_splits.items() if sp.get("train")
    }
    if not train_sources:
        raise ValueError("Aucune source avec un pool train pour calibration.")

    n_sources = len(train_sources)
    per_source = max(1, n_total // n_sources)

    rng = random.Random(seed + 1)  # décorrélé du seed du train sampler
    calib: list[UnifiedSample] = []
    sizes_used: dict[str, int] = {}

    for name, pool in sorted(train_sources.items()):
        # tirage sans remise sur indices : permet de retirer ensuite du pool
        k = min(per_source, len(pool))
        if k == 0:
            continue
        picked_idx = set(rng.sample(range(len(pool)), k))
        picks = [pool[i] for i in picked_idx]
        calib.extend(picks)
        sizes_used[name] = k

        if remove_from_pools:
            # on remplace la liste train par la sous-liste sans les indices
            # calibrés — garantit disjonction stricte calibration ⟂ train.
            source_splits[name]["train"] = [
                s for i, s in enumerate(pool) if i not in picked_idx
            ]

    # Shuffle global pour éviter que les N premiers samples soient tous de
    # la même source (biaiserait la stat d'activation côté AWQ/GPTQ).
    rng.shuffle(calib)

    print(f"  [calibration] {len(calib)} samples stratifiés "
          f"(~{per_source}/source sur {n_sources} sources)"
          + (" [retirés du pool train]" if remove_from_pools else ""))
    for name, n in sorted(sizes_used.items()):
        print(f"    - {name:<18}: {n}")

    return calib


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def describe_mix(
    sizes: dict[str, int],
    mix_cfg: MixConfig,
    n_total: int | None = None,
) -> str:
    """Tableau texte lisible des poids + tailles effectives."""
    w = compute_weights(sizes, mix_cfg)
    rows = [f"{'dataset':<16}{'n_avail':>10}{'weight':>10}{'n_sampled':>12}"]
    for name in sorted(sizes, key=lambda k: -w[k]):
        n = sizes[name]
        wi = w[name]
        ns = int(round(n_total * wi)) if n_total else "-"
        rows.append(f"{name:<16}{n:>10}{wi:>10.4f}{str(ns):>12}")
    return "\n".join(rows)
