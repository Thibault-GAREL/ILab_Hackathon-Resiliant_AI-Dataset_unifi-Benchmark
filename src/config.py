"""
Resilient AI Challenge — configuration des datasets et des stratégies de mix.

Source de vérité pour :
- les identifiants HuggingFace de chaque dataset,
- les noms de splits natifs,
- le type de tâche (vqa / captioning / multiple_choice),
- la métrique canonique de chaque dataset.

Tout module consommateur (loaders, mix, benchmark) doit lire ici — pas de
constantes en doublon ailleurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TaskType = Literal["vqa", "captioning", "multiple_choice"]
MetricType = Literal[
    "anls",                    # DocVQA (Average Normalized Levenshtein Similarity)
    "vqa_accuracy",            # VQAv2, TextVQA, OK-VQA (10-answer consensus)
    "relaxed_accuracy",        # ChartQA (tolérance 5% sur valeurs numériques)
    "multiple_choice",         # MMMU, AI2D
    "realworld_accuracy",      # RealWorldQA (exact match ou MC selon sample)
    "cider",                   # COCO Captions
]


@dataclass(frozen=True)
class DatasetSpec:
    """Description d'un dataset source.

    - `hf_id` et `config` pointent vers HuggingFace Hub.
    - `splits` mappe nos noms canoniques (train/val/test) vers le split natif.
      Si un split n'existe pas côté Hub (ex: test labels non publics pour
      VQAv2), on laisse None et on applique le split 80/10/10 interne à
      partir du train.
    - `task` et `metric` pilotent schema/benchmark.
    - `tier` : 1 = indispensable, 2 = couverture.
    """
    name: str
    hf_id: str
    config: str | None
    splits: dict[str, str | None]
    task: TaskType
    metric: MetricType
    tier: int
    notes: str = ""


# ---- Tier 1 : indispensables pour le challenge ------------------------------

DOCVQA = DatasetSpec(
    name="docvqa",
    hf_id="pixparse/docvqa-single-page-questions",
    config=None,
    # pixparse : Parquet-native, 39.5k train / 5.3k val / 5.2k test avec TOUS
    # les labels (contrairement à lmms-lab/DocVQA dont le test a labels cachés).
    splits={"train": "train", "val": "validation", "test": "test"},
    task="vqa",
    metric="anls",
    tier=1,
    notes="pixparse clean : train+val+test avec labels. Parquet-native (pas de loading script). Schema: image, question, answers, question_id, ocr_results, other_metadata.",
)

TEXTVQA = DatasetSpec(
    name="textvqa",
    hf_id="lmms-lab/textvqa",
    config=None,
    splits={"train": "train", "val": "validation", "test": None},
    task="vqa",
    metric="vqa_accuracy",
    tier=1,
    notes="lmms-lab repo Parquet-native (pas de loading script cassé en datasets>=3.0). 34.6k train / 5k val / 5.7k test (labels test cachés). 10 réponses humaines.",
)

MMMU = DatasetSpec(
    name="mmmu",
    hf_id="lmms-lab/MMMU",
    config=None,
    # lmms-lab/MMMU unifie les 30 configs disciplinaires en un seul repo
    # Parquet. Plus besoin de boucler sur get_dataset_config_names.
    # Test answers released 2026-02-12 → utilisables pour benchmark.
    splits={"train": "dev", "val": "validation", "test": "test"},
    task="multiple_choice",
    metric="multiple_choice",
    tier=1,
    notes="lmms-lab unifié (30 disciplines en un repo). 150 dev / 900 val / 10.5k test. Mix MC + open-ended. Colonnes: id, question, options, answer, image_1..7, subfield.",
)

VQAV2 = DatasetSpec(
    name="vqav2",
    hf_id="lmms-lab/VQAv2",
    config=None,
    # lmms-lab/VQAv2 Parquet-native (remplace HuggingFaceM4/VQAv2 cassé par
    # datasets>=3.0). 214k val / 107k testdev / 448k test, pas de train public
    # sur ce repo : on utilise 'validation' comme pool train (suffit largement
    # avec max_per_source=150k).
    splits={"train": "validation", "val": None, "test": "testdev"},
    task="vqa",
    metric="vqa_accuracy",
    tier=1,
    notes="lmms-lab Parquet. Pas de train public → validation=pool train (214k), testdev=eval. 80/10/10 fallback pour val/test si besoin. Consensus 10 réponses.",
)


# ---- Tier 2 : couverture large ---------------------------------------------

CHARTQA = DatasetSpec(
    name="chartqa",
    hf_id="HuggingFaceM4/ChartQA",
    config=None,
    splits={"train": "train", "val": "val", "test": "test"},
    task="vqa",
    metric="relaxed_accuracy",
    tier=2,
    notes="Questions sur graphiques. Métrique tolérance ±5% sur numériques.",
)

AI2D = DatasetSpec(
    name="ai2d",
    hf_id="lmms-lab/ai2d",
    config=None,
    # lmms-lab/ai2d n'expose que le split test (~3k samples).
    # On utilise test comme pool unique et on split 80/10/10 en interne.
    splits={"train": "test", "val": None, "test": None},
    task="multiple_choice",
    metric="multiple_choice",
    tier=2,
    notes="Pas de train public sur lmms-lab/ai2d (eval-only). Fallback 80/10/10 sur le split test (~3k). Pour full train, bascule sur ai2d/ai2d-rc.",
)

REALWORLDQA = DatasetSpec(
    name="realworldqa",
    hf_id="xai-org/RealworldQA",
    config=None,
    splits={"train": None, "val": None, "test": "test"},
    task="vqa",
    metric="realworld_accuracy",
    tier=2,
    notes="Eval-only (765 samples). Utilisé uniquement pour benchmark, pas train.",
)

COCO_CAPTIONS = DatasetSpec(
    name="coco_captions",
    hf_id="lmms-lab/COCO-Caption",
    config=None,
    # lmms-lab/COCO-Caption : Parquet-native avec images EMBED (vs yerevann qui
    # ne contient que des URLs). Pas de train officiel → on utilise 'val' (40.5k)
    # comme pool train + 'test' (40.8k) comme pool test. Val interne reconstruite
    # par fallback 80/10/10 dans mix.py.
    splits={"train": "val", "val": None, "test": "test"},
    task="captioning",
    metric="cider",
    tier=2,
    notes="lmms-lab Parquet natif (remplace yerevann/coco-karpathy qui n'embed pas les images). Schema: image, answer (list captions), question_id, file_name, coco_url. Prompt 'Describe this image.' ajouté côté loader.",
)

OK_VQA = DatasetSpec(
    name="ok_vqa",
    hf_id="Multimodal-Fatima/OK-VQA_train",
    config=None,
    splits={"train": "train", "val": None, "test": None},  # test chargé séparément
    task="vqa",
    metric="vqa_accuracy",
    tier=2,
    notes="Test set sur dataset séparé 'OK-VQA_test'. ~14k train, ~5k test.",
)


# ---- v0.3 : couverture étendue (raisonnement CoT + infographies + math) ----

SCIENCEQA = DatasetSpec(
    name="scienceqa",
    hf_id="derek-thomas/ScienceQA",
    config=None,
    # Parquet-native, Image() feature (bytes embed confirmé via /api/datasets/.../parquet).
    # 12.7k train / 4.24k val / 4.24k test. ~48% des samples ont une image — le loader
    # filtre les text-only (hors scope image→text).
    splits={"train": "train", "val": "validation", "test": "test"},
    task="multiple_choice",
    metric="multiple_choice",
    tier=2,
    notes="Raisonnement multi-step K-12. Champs lecture/solution conservés en metadata pour CoT distillation futur. MC format (answer=int8 index dans choices 2-5).",
)

INFOGRAPHICVQA = DatasetSpec(
    name="infographicvqa",
    hf_id="lmms-lab/DocVQA",
    config="InfographicVQA",
    # lmms-lab/DocVQA expose 2 configs : DocVQA (qu'on n'utilise pas ici, on a pixparse)
    # et InfographicVQA (6.09k val uniquement, pas de train natif).
    # Fallback 80/10/10 interne → ~4.8k train fallback + 600 val + 600 test.
    splits={"train": "validation", "val": None, "test": None},
    task="vqa",
    metric="anls",
    tier=2,
    notes="Infographies complexes (layout + OCR dense + visualisations). Complémente DocVQA (docs classiques) en stressant l'OCR multi-échelle. Métrique ANLS cohérente avec DocVQA.",
)

MATHVISTA = DatasetSpec(
    name="mathvista",
    hf_id="AI4Math/MathVista",
    config=None,
    # testmini (1k) = seul split avec labels publics. test (5.14k) = labels scellés
    # (benchmark à soumission). Fallback 80/10/10 sur testmini → 800 train / 100 val / 100 test.
    splits={"train": "testmini", "val": None, "test": None},
    task="vqa",  # mix MC + free_form géré dans le loader
    metric="vqa_accuracy",
    tier=2,
    notes="Math multimodal. Colonne `decoded_image` pour les bytes. Mix MC (answer∈choices) + free_form (text/int/float). Pool petit mais signal math indispensable vs MMMU math-only.",
)


ALL_DATASETS: dict[str, DatasetSpec] = {
    d.name: d
    for d in (
        DOCVQA, TEXTVQA, MMMU, VQAV2, CHARTQA, AI2D, REALWORLDQA, COCO_CAPTIONS, OK_VQA,
        SCIENCEQA, INFOGRAPHICVQA, MATHVISTA,
    )
}

TIER1 = [d.name for d in ALL_DATASETS.values() if d.tier == 1]
TIER2 = [d.name for d in ALL_DATASETS.values() if d.tier == 2]


# ---- Stratégies de mix ------------------------------------------------------

MixStrategy = Literal["uniform", "sqrt", "task_balanced", "temperature"]


@dataclass(frozen=True)
class MixConfig:
    """Paramètres d'un mix de datasets.

    - `strategy` : uniform | sqrt | task_balanced | temperature
    - `temperature` : utilisé uniquement si strategy == 'temperature'
    - `max_per_dataset` : cap éventuel (None = pas de cap)
    - `seed` : pour la reproductibilité — DOIT être le même pour toute
      l'équipe pour que les benchmarks soient comparables.
    """
    strategy: MixStrategy = "sqrt"
    temperature: float = 2.0
    max_per_dataset: int | None = None
    seed: int = 42


# ---- Splits -----------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    """Découpage train/val/test.

    - `use_official_splits` : si True, on utilise les splits natifs quand ils
      existent (recommandé pour la comparabilité). Sinon 80/10/10 interne.
    - `ratios` : utilisé uniquement si use_official_splits=False ou si un
      dataset n'a pas de split officiel (ex: OK-VQA train uniquement).
    - `seed` : idem MixConfig — reproductibilité stricte.
    """
    use_official_splits: bool = True
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42


# ---- Identifiants de publication -------------------------------------------

# Repo HuggingFace cible par défaut (username Mathis). À changer si la team
# bascule vers un repo "org" commun plus tard.
DEFAULT_HUB_REPO = "Mathos34400/resilient-ai-unified"
DATASET_VERSION = "v0.3.0"  # v0.3.0 : +ScienceQA +InfographicVQA +MathVista + calibration split (AWQ/GPTQ)
