"""
Métriques par type de dataset.

Implémenté à la main pour rester indépendant de pycocoevalcap (dépendance
lourde) sur les métriques courantes. `evaluate` HF est utilisé pour CIDEr /
BLEU uniquement (seul cas où une impl. de qualité est non triviale).

Référence canonique par métrique :
- ANLS (DocVQA)                : https://arxiv.org/abs/2007.00398
- VQA accuracy (VQAv2/TextVQA) : Antol et al. 2015, 10-answer consensus
- Relaxed accuracy (ChartQA)   : Masry et al. 2022, tolérance 5% numérique
- Multiple choice              : accuracy top-1 lettre
- CIDEr (COCO Captions)        : Vedantam et al. 2015
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

try:
    import Levenshtein  # python-Levenshtein
except ImportError:  # fallback pur Python
    Levenshtein = None


# ----------------------------------------------------------------------------
# Normalisation texte commune VQA
# ----------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_CONTRACTIONS = {
    "dont": "don't", "cant": "can't", "wont": "won't", "isnt": "isn't",
    "arent": "aren't", "wasnt": "wasn't", "werent": "weren't",
}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_answer(s: str) -> str:
    """Normalisation standard VQA : lowercase, no punct, no articles, contractions."""
    s = str(s).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    tokens = [t for t in s.split() if t not in _ARTICLES]
    tokens = [_CONTRACTIONS.get(t, t) for t in tokens]
    tokens = [_NUMBER_WORDS.get(t, t) for t in tokens]
    return " ".join(tokens)


# ----------------------------------------------------------------------------
# VQA accuracy (consensus 10 réponses)
# ----------------------------------------------------------------------------

def vqa_accuracy(pred: str, gt_answers: Sequence[str]) -> float:
    """Score VQAv2 : min(# matches / 3, 1).

    Si moins de 10 gt fournies (TextVQA/OK-VQA souvent 10, mais pas garanti),
    on fait au mieux avec ce qu'on a.
    """
    if not gt_answers:
        return 0.0
    p = normalize_answer(pred)
    norm_gt = [normalize_answer(a) for a in gt_answers]
    # VQAv2 officiel : pour chaque annotateur, on compte les matches parmi les
    # 9 autres, puis on moyenne. Simplification ici (équivalent sur 10 gt) :
    matches = sum(1 for a in norm_gt if a == p)
    return min(matches / 3.0, 1.0)


# ----------------------------------------------------------------------------
# ANLS (DocVQA)
# ----------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if Levenshtein is not None:
        return Levenshtein.distance(a, b)
    # fallback DP
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[n]


def anls(pred: str, gt_answers: Sequence[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA)."""
    if not gt_answers:
        return 0.0
    p = str(pred).lower().strip()
    best = 0.0
    for gt in gt_answers:
        g = str(gt).lower().strip()
        if not g:
            continue
        d = _edit_distance(p, g)
        nl = 1.0 - d / max(len(p), len(g), 1)
        if nl >= threshold:
            best = max(best, nl)
    return best


# ----------------------------------------------------------------------------
# Relaxed accuracy (ChartQA)
# ----------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"-?\d+(?:[.,]\d+)?%?")


def _extract_number(s: str) -> float | None:
    m = _FLOAT_RE.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", ".").rstrip("%")
    try:
        return float(raw)
    except ValueError:
        return None


def relaxed_accuracy(pred: str, gt_answers: Sequence[str], tol: float = 0.05) -> float:
    """ChartQA : tolérance 5% si numérique, exact match normalisé sinon."""
    if not gt_answers:
        return 0.0
    p_num = _extract_number(pred)
    for gt in gt_answers:
        g_num = _extract_number(gt)
        if p_num is not None and g_num is not None:
            if g_num == 0:
                return 1.0 if abs(p_num) < 1e-6 else 0.0
            if abs(p_num - g_num) / max(abs(g_num), 1e-6) <= tol:
                return 1.0
        else:
            if normalize_answer(pred) == normalize_answer(gt):
                return 1.0
    return 0.0


# ----------------------------------------------------------------------------
# Multiple choice
# ----------------------------------------------------------------------------

def multiple_choice_accuracy(pred: str, correct_letter: str) -> float:
    """Extrait la lettre-réponse de `pred` et compare à `correct_letter`.

    Accepté :
      - pred exactement la lettre (éventuellement avec point)   : "B", "B."
      - forme parenthésée                                        : "(B)"
      - tournure explicite                                       : "the answer is B"
      - lettre en fin de phrase suivie de ponctuation            : "... B."
      - lettre isolée entre séparateurs                          : " B ", ", B,"
      - fallback : première lettre A-H trouvée

    L'ordre des patterns est crucial : commencer par les plus spécifiques
    pour éviter que "The answer is B" ne renvoie 'T'.
    """
    p = str(pred).strip().upper()
    if not p:
        return 0.0
    target = correct_letter.upper()

    # Cas trivial : réponse == exactement la lettre
    if p == target or p == target + ".":
        return 1.0

    # Patterns spécifiques prioritaires
    patterns = [
        r"\(([A-H])\)",                           # (B)
        r"ANSWER\s+IS\s+([A-H])\b",               # the answer is B
        r"\b([A-H])\s*\.\s*$",                    # finit par 'B.'
        r"\b([A-H])\s*\)",                        # 'B)'
        r"(?:^|[\s,;:])([A-H])(?:[\s,;:.\)]|$)",  # lettre isolée
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return 1.0 if m.group(1) == target else 0.0

    # Dernier fallback : première lettre A-H rencontrée
    m = re.search(r"[A-H]", p)
    if m:
        return 1.0 if m.group(0) == target else 0.0
    return 0.0


# ----------------------------------------------------------------------------
# RealWorldQA : soit MC (lettre dans pred), soit exact match normalisé
# ----------------------------------------------------------------------------

def realworld_accuracy(pred: str, gt_answers: Sequence[str]) -> float:
    if not gt_answers:
        return 0.0
    gt = gt_answers[0]
    if len(gt) == 1 and gt.isalpha():
        return multiple_choice_accuracy(pred, gt)
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


# ----------------------------------------------------------------------------
# CIDEr via `evaluate` (chargé à la demande)
# ----------------------------------------------------------------------------

_cider_cache = None


def cider_corpus(preds: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    """CIDEr au niveau corpus (pas par sample).

    Charger une seule fois — l'import est lourd.
    """
    global _cider_cache
    if _cider_cache is None:
        try:
            import evaluate  # type: ignore
            _cider_cache = evaluate.load("cider")
        except Exception as e:
            # Fallback : retourner une approximation via BLEU-4 faute de mieux,
            # avec avertissement.
            print(f"[metrics] CIDEr indisponible ({e}) ; fallback BLEU-4 corpus.")
            import evaluate  # type: ignore
            bleu = evaluate.load("bleu")
            res = bleu.compute(
                predictions=list(preds),
                references=[list(r) for r in refs],
                max_order=4,
            )
            return float(res["bleu"])
    res = _cider_cache.compute(
        predictions=list(preds), references=[list(r) for r in refs]
    )
    # evaluate.load('cider') renvoie {'CIDEr': value}
    return float(next(iter(res.values())))


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

def score_sample(metric_hint: str, pred: str, sample: dict) -> float:
    """Dispatch pour métriques par-sample (tout sauf CIDEr)."""
    if metric_hint == "vqa_accuracy":
        return vqa_accuracy(pred, sample["answers"])
    if metric_hint == "anls":
        return anls(pred, sample["answers"])
    if metric_hint == "relaxed_accuracy":
        return relaxed_accuracy(pred, sample["answers"])
    if metric_hint == "multiple_choice":
        letter = chr(65 + sample["correct_choice_idx"])
        return multiple_choice_accuracy(pred, letter)
    if metric_hint == "realworld_accuracy":
        return realworld_accuracy(pred, sample["answers"])
    if metric_hint == "cider":
        # CIDEr est corpus-level : on ne score pas ici, on retourne NaN.
        return float("nan")
    raise ValueError(f"metric_hint inconnu : {metric_hint}")
