"""
Loaders par dataset source → UnifiedSample.

Chaque loader :
1. charge via `datasets.load_dataset`,
2. itère sur les samples et produit des UnifiedSample via adaptation du
   schéma natif,
3. retourne un iterator (pour composition avec le mix sampler).

Mode streaming : si `max_samples` est passé, on active `streaming=True` et
on stoppe après N samples — indispensable pour éviter de télécharger 20 GB
de VQAv2 quand on n'a besoin que de 500 samples (ex: build sandbox, CI).
Sur Runpod avec disque large, laisser `max_samples=None` charge tout.

Les images sont converties en **bytes JPEG** dès l'étape load (au lieu de
rester en PIL.Image avec pixels décompressés). Gain de RAM typique : 5-20×
(une image 512×512 RGB = ~800 KB en PIL, ~50-100 KB en JPEG). Sans ça,
accumuler 100-200k samples en RAM Runpod (100-120 GB) casse vite.
"""
from __future__ import annotations

import itertools
import json
from io import BytesIO
from typing import Iterator, Callable
from PIL import Image

from src.config import DatasetSpec, ALL_DATASETS
from src.data.schema import UnifiedSample


# ---- Helper chargement streaming ou non -----------------------------------

def _iter_rows(hf_id: str, config: str | None, split: str, max_samples: int | None):
    """Itère sur les rows d'un dataset HF, streaming si `max_samples` est set.

    On préfère streaming quand possible car :
    - évite de télécharger tout le dataset pour n'en garder qu'une fraction,
    - consomme beaucoup moins de RAM / disque sur VQAv2 / COCO / DocVQA.

    Note : depuis `datasets>=4.0`, le paramètre `trust_remote_code` a été
    retiré. Tous nos repos ont été migrés vers des variantes Parquet-native
    (lmms-lab/VQAv2, lmms-lab/textvqa, lmms-lab/MMMU, lmms-lab/DocVQA,
    pixparse/docvqa-single-page-questions, derek-thomas/ScienceQA, etc.)
    donc aucun besoin de builder script custom.
    """
    from datasets import load_dataset
    if max_samples is None:
        ds = (
            load_dataset(hf_id, config, split=split)
            if config
            else load_dataset(hf_id, split=split)
        )
        return iter(ds)
    ds = (
        load_dataset(hf_id, config, split=split, streaming=True)
        if config
        else load_dataset(hf_id, split=split, streaming=True)
    )
    return itertools.islice(ds, max_samples)


# ---- Helpers ----------------------------------------------------------------

import os

# Resize (côté plus long) + encodage JPEG forcés pour plafonner la RAM.
# Défaut : 512 px (valeur utile côté Gemma, compatible processor 256/512/768).
# Pour désactiver totalement : export MAX_IMAGE_DIM=0.
# DocVQA peut stocker des pages multi-Mo en full-res, ce qui explose en RAM
# quand on accumule 100-200k samples avant écriture disque.
_MAX_IMAGE_DIM_ENV = os.environ.get("MAX_IMAGE_DIM", "512").strip()
_max_dim_val = int(_MAX_IMAGE_DIM_ENV) if _MAX_IMAGE_DIM_ENV else 512
MAX_IMAGE_DIM: int | None = _max_dim_val if _max_dim_val > 0 else None

# Qualité JPEG : 85 = bon compromis taille/qualité, standard (même param que
# datasets.Image quand elle sérialise pour Parquet).
_JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))


def _ensure_image_bytes(img) -> bytes:
    """Convertit toute représentation d'image HF en bytes JPEG compressés.

    Gain RAM : en PIL.Image RGB 512×512 les pixels décompressés font ~800 KB,
    ici on stocke ~50-100 KB par image. Sur 200k samples ça fait la différence
    entre un build qui tient dans 12 GB ou un qui crashe à 119 GB.

    HF renvoie parfois déjà un PIL.Image, parfois un dict {bytes, path}.
    On normalise vers bytes JPEG.
    """
    # 1. Récupère un PIL.Image
    if isinstance(img, Image.Image):
        pil = img
    elif isinstance(img, dict) and img.get("bytes"):
        pil = Image.open(BytesIO(img["bytes"]))
    elif isinstance(img, dict) and img.get("path"):
        pil = Image.open(img["path"])
    elif isinstance(img, (bytes, bytearray)):
        pil = Image.open(BytesIO(img))
    else:
        raise ValueError(f"Image non reconnue : {type(img)}")

    # 2. Resize si nécessaire (plafond côté long)
    if MAX_IMAGE_DIM is not None:
        w, h = pil.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / float(max(w, h))
            pil = pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    # 3. Convertit en RGB (JPEG ne supporte pas RGBA / P / L directement bien)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    # 4. Encode en JPEG bytes
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=False)
    return buf.getvalue()


# Backward-compat alias (certains tests peuvent encore appeler _ensure_pil).
_ensure_pil = _ensure_image_bytes


def _sid(source: str, idx: int, suffix: str = "") -> str:
    base = f"{source}:{idx}"
    return f"{base}:{suffix}" if suffix else base


# ---- VQAv2 ------------------------------------------------------------------

def load_vqav2(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    # lmms-lab/VQAv2 : schema identique — image, question, answers (list 10), image_id,
    # question_id, question_type, answer_type, multiple_choice_answer.
    for i, row in enumerate(_iter_rows("lmms-lab/VQAv2", None, split, max_samples)):
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            continue  # test sans labels
        # lmms-lab renvoie parfois list[dict{answer, answer_confidence}] — on aplatit
        if isinstance(answers[0], dict):
            answers = [a.get("answer", "") for a in answers if a.get("answer")]
        if not answers:
            continue
        # Canonique : multiple_choice_answer en tête si dispo (1 réponse majoritaire)
        canon = row.get("multiple_choice_answer")
        if canon and canon not in answers:
            answers = [canon] + answers
        elif canon:
            answers = [canon] + [a for a in answers if a != canon]
        yield UnifiedSample(
            id=_sid("vqav2", i),
            source="vqav2",
            task="vqa",
            image=_ensure_pil(row["image"]),
            question=row["question"],
            answers=answers,
            metric_hint="vqa_accuracy",
            metadata={
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id"),
                "question_type": row.get("question_type"),
                "answer_type": row.get("answer_type"),
            },
        )


# ---- TextVQA ----------------------------------------------------------------

def load_textvqa(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    # lmms-lab/textvqa : Parquet, schema identique à facebook/textvqa.
    for i, row in enumerate(_iter_rows("lmms-lab/textvqa", None, split, max_samples)):
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            continue  # test labels cachés
        yield UnifiedSample(
            id=_sid("textvqa", i),
            source="textvqa",
            task="vqa",
            image=_ensure_pil(row["image"]),
            question=row["question"],
            answers=answers,
            metric_hint="vqa_accuracy",
            metadata={
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id"),
            },
        )


# ---- DocVQA -----------------------------------------------------------------

def load_docvqa(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    # pixparse/docvqa-single-page-questions : Parquet, TOUS splits avec labels.
    # Schema: image, question, question_id, answers (list), data_split, ocr_results, other_metadata.
    for i, row in enumerate(_iter_rows("pixparse/docvqa-single-page-questions", None, split, max_samples)):
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            continue
        # other_metadata peut contenir docId / page / etc. — on le sérialise tel quel
        om = row.get("other_metadata") or {}
        if isinstance(om, str):
            try:
                om = json.loads(om)
            except Exception:
                om = {"_raw": om}
        yield UnifiedSample(
            id=_sid("docvqa", i),
            source="docvqa",
            task="vqa",
            image=_ensure_pil(row["image"]),
            question=row["question"],
            answers=answers,
            metric_hint="anls",
            metadata={
                "question_id": row.get("question_id"),
                "data_split": row.get("data_split"),
                # on ne garde PAS ocr_results (lourd, pas utile pour train image→text)
                **({"doc_id": om.get("ucsf_document_id")} if isinstance(om, dict) and om.get("ucsf_document_id") else {}),
            },
        )


# ---- MMMU -------------------------------------------------------------------

def load_mmmu(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    """lmms-lab/MMMU : un seul repo Parquet unifié sur les 30 disciplines.

    Colonnes : id, question, options, answer, image_1..image_7, subfield,
    question_type (multiple-choice | open).
    Plus besoin d'itérer sur 30 configs comme avec MMMU/MMMU.
    """
    for i, row in enumerate(_iter_rows("lmms-lab/MMMU", None, split, max_samples)):
        options = row.get("options")
        if isinstance(options, str):
            try:
                options = json.loads(options.replace("'", '"'))
            except Exception:
                options = None

        question_type = (row.get("question_type") or "").lower()
        correct = (row.get("answer") or "").strip()

        # Première image non-nulle dans image_1..image_7 (la majorité n'en a qu'une)
        img = None
        for k in ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"):
            v = row.get(k)
            if v is not None:
                img = v
                break
        if img is None:
            continue  # échantillon text-only, hors scope image→text
        img_bytes = _ensure_pil(img)

        mmmu_id = row.get("id") or _sid("mmmu", i, str(row.get("subfield", "na")).replace(" ", "_"))

        if question_type == "multiple-choice" and options:
            try:
                idx = ord(correct.upper()) - ord("A")
            except Exception:
                continue
            if not (0 <= idx < len(options)):
                continue
            yield UnifiedSample(
                id=str(mmmu_id),
                source="mmmu",
                task="multiple_choice",
                image=img_bytes,
                question=row["question"],
                answers=[correct.upper()],
                choices=options,
                correct_choice_idx=idx,
                metric_hint="multiple_choice",
                metadata={
                    "subfield": row.get("subfield"),
                    "topic_difficulty": row.get("topic_difficulty"),
                    "question_type": "multiple-choice",
                },
            )
        else:
            if not correct:
                continue
            yield UnifiedSample(
                id=str(mmmu_id),
                source="mmmu",
                task="vqa",
                image=img_bytes,
                question=row["question"],
                answers=[correct],
                metric_hint="vqa_accuracy",
                metadata={
                    "subfield": row.get("subfield"),
                    "question_type": "open",
                },
            )


# ---- ChartQA ----------------------------------------------------------------

def load_chartqa(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    for i, row in enumerate(_iter_rows("HuggingFaceM4/ChartQA", None, split, max_samples)):
        label = row.get("label") or row.get("answer")
        if not label:
            continue
        answers = [label] if isinstance(label, str) else list(label)
        yield UnifiedSample(
            id=_sid("chartqa", i),
            source="chartqa",
            task="vqa",
            image=_ensure_pil(row["image"]),
            question=row["query"] if "query" in row else row["question"],
            answers=answers,
            metric_hint="relaxed_accuracy",
            metadata={"human_or_machine": row.get("human_or_machine")},
        )


# ---- AI2D -------------------------------------------------------------------

def load_ai2d(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    for i, row in enumerate(_iter_rows("lmms-lab/ai2d", None, split, max_samples)):
        options = row.get("options") or [
            row.get(f"option_{j}") for j in range(1, 5) if row.get(f"option_{j}")
        ]
        options = [o for o in options if o]
        correct_raw = row.get("answer") or row.get("correct_answer")
        if not options or correct_raw is None:
            continue
        # correct_answer peut être lettre ou index selon la variante
        if isinstance(correct_raw, str) and correct_raw.isalpha():
            idx = ord(correct_raw.upper()) - ord("A")
        else:
            try:
                idx = int(correct_raw)
            except Exception:
                continue
        if not (0 <= idx < len(options)):
            continue
        yield UnifiedSample(
            id=_sid("ai2d", i),
            source="ai2d",
            task="multiple_choice",
            image=_ensure_pil(row["image"]),
            question=row["question"],
            answers=[chr(65 + idx)],
            choices=options,
            correct_choice_idx=idx,
            metric_hint="multiple_choice",
            metadata={},
        )


# ---- RealWorldQA ------------------------------------------------------------

def load_realworldqa(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    for i, row in enumerate(_iter_rows("xai-org/RealworldQA", None, split, max_samples)):
        answer = row.get("answer") or row.get("correct_answer")
        if not answer:
            continue
        # Certaines questions contiennent déjà les choix inline — on ne tente pas
        # de les re-parser ; on traite en VQA ouvert.
        yield UnifiedSample(
            id=_sid("realworldqa", i),
            source="realworldqa",
            task="vqa",
            image=_ensure_pil(row["image"]),
            question=row["question"],
            answers=[str(answer).strip()],
            metric_hint="realworld_accuracy",
            metadata={},
        )


# ---- COCO Captions (adapté en Q/R) -----------------------------------------

COCO_CAPTION_PROMPT = "Describe this image."


def load_coco_captions(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    """lmms-lab/COCO-Caption : Parquet natif, images embed.

    Schema : image (PIL), answer (list[str]=5 captions), question_id,
    id, license, file_name, coco_url, height, width, date_captured.
    Le champ 'question' (souvent vide) est remplacé par un prompt fixe
    "Describe this image." pour transformer le captioning en Q/R cohérent
    avec les autres sources VQA.
    """
    for i, row in enumerate(_iter_rows("lmms-lab/COCO-Caption", None, split, max_samples)):
        # 'answer' est la liste des captions (5 en général)
        captions = row.get("answer") or row.get("answers") or []
        if isinstance(captions, str):
            captions = [captions]
        # filtre les None / vides
        captions = [c for c in captions if c]
        if not captions:
            continue
        img = row.get("image")
        if img is None:
            continue
        yield UnifiedSample(
            id=_sid("coco_captions", i),
            source="coco_captions",
            task="captioning",
            image=_ensure_pil(img),
            question=COCO_CAPTION_PROMPT,
            answers=captions,  # toutes les références (utilisé par CIDEr)
            metric_hint="cider",
            metadata={
                "question_id": row.get("question_id"),
                "image_id": row.get("id"),
                "file_name": row.get("file_name"),
            },
        )


# ---- OK-VQA -----------------------------------------------------------------

def load_okvqa(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    # Le repo utilisé par la communauté sépare train et test en deux datasets.
    hf_id = "Multimodal-Fatima/OK-VQA_train" if split == "train" else "Multimodal-Fatima/OK-VQA_test"
    # le split interne s'appelle 'train' dans les deux cas
    for i, row in enumerate(_iter_rows(hf_id, None, "train", max_samples)):
        answers = row.get("answers") or row.get("raw_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            continue
        yield UnifiedSample(
            id=_sid("ok_vqa", i, split),
            source="ok_vqa",
            task="vqa",
            image=_ensure_pil(row["image"]),
            question=row["question"],
            answers=answers,
            metric_hint="vqa_accuracy",
            metadata={"question_id": row.get("question_id")},
        )


# ---- v0.3 : ScienceQA -------------------------------------------------------

def load_scienceqa(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    """derek-thomas/ScienceQA : MC format, answer=int8 index dans choices.

    Filtrage obligatoire : ~52% des samples sont text-only (pas d'image),
    hors scope pour notre pipeline image→text — on les skip.

    Métadonnées conservées : subject/topic/grade + lecture/solution tronqués
    à 500 chars (utile pour une future distillation CoT où le target serait
    solution au lieu de la lettre).
    """
    for i, row in enumerate(_iter_rows("derek-thomas/ScienceQA", None, split, max_samples)):
        img = row.get("image")
        if img is None:
            continue  # text-only : skip
        choices = row.get("choices") or []
        if not choices:
            continue
        idx = row.get("answer")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(choices)):
            continue

        question = (row.get("question") or "").strip()
        hint = (row.get("hint") or "").strip()
        # Le hint apporte souvent un contexte essentiel (ex: "Regarde cette carte.")
        # On le préfixe à la question pour que le modèle le voie à l'inférence.
        if hint:
            question = f"{hint}\n\n{question}" if question else hint

        yield UnifiedSample(
            id=_sid("scienceqa", i, split),
            source="scienceqa",
            task="multiple_choice",
            image=_ensure_pil(img),
            question=question,
            answers=[chr(65 + idx)],
            choices=choices,
            correct_choice_idx=idx,
            metric_hint="multiple_choice",
            metadata={
                "subject": row.get("subject"),
                "topic": row.get("topic"),
                "grade": row.get("grade"),
                "task_type": row.get("task"),
                # Tronqués pour ne pas gonfler la metadata JSON (impacte le Parquet)
                "lecture": (row.get("lecture") or "")[:500],
                "solution": (row.get("solution") or "")[:500],
            },
        )


# ---- v0.3 : InfographicVQA --------------------------------------------------

def load_infographicvqa(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    """lmms-lab/DocVQA, config 'InfographicVQA' : val 6.09k (split unique natif).

    Schema identique à DocVQA : questionId, question, image, answers (list),
    question_types, docId, data_split. Metric ANLS (edit distance).
    Les images sont des infographies grand format avec texte dense multi-échelle.
    """
    for i, row in enumerate(_iter_rows("lmms-lab/DocVQA", "InfographicVQA", split, max_samples)):
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answers = [a for a in answers if a]
        if not answers:
            continue
        img = row.get("image")
        if img is None:
            continue
        yield UnifiedSample(
            id=_sid("infographicvqa", i),
            source="infographicvqa",
            task="vqa",
            image=_ensure_pil(img),
            question=row.get("question") or "",
            answers=answers,
            metric_hint="anls",
            metadata={
                "question_id": row.get("questionId"),
                "doc_id": row.get("docId"),
                "question_types": row.get("question_types"),
            },
        )


# ---- v0.3 : MathVista -------------------------------------------------------

def load_mathvista(split: str, max_samples: int | None = None) -> Iterator[UnifiedSample]:
    """AI4Math/MathVista : testmini (1k, labels publics).

    Schema mixte via question_type ∈ {multi_choice, free_form} :
    - MC : answer est une string qui correspond à une entrée de `choices`.
      On l'indexe pour peupler correct_choice_idx.
    - free_form : answer est text/int/float, traité en VQA ouvert.

    Colonne `decoded_image` contient les bytes de l'image (pas `image` qui est
    juste un filename).
    """
    for i, row in enumerate(_iter_rows("AI4Math/MathVista", None, split, max_samples)):
        img = row.get("decoded_image") or row.get("image")
        if img is None:
            continue
        # Skip si l'image est un string (filename path) et pas un objet image
        if isinstance(img, str):
            continue
        answer = row.get("answer")
        if answer is None or answer == "":
            continue
        answer = str(answer).strip()
        question = (row.get("question") or row.get("query") or "").strip()
        if not question:
            continue

        question_type = (row.get("question_type") or "").lower()
        choices = row.get("choices") or []

        if question_type == "multi_choice" and choices:
            # Trouver l'index de la réponse dans choices (insensible à la casse)
            idx = None
            for j, c in enumerate(choices):
                if str(c).strip() == answer or str(c).strip().lower() == answer.lower():
                    idx = j
                    break
            if idx is not None:
                yield UnifiedSample(
                    id=_sid("mathvista", i, str(row.get("pid") or "")),
                    source="mathvista",
                    task="multiple_choice",
                    image=_ensure_pil(img),
                    question=question,
                    answers=[chr(65 + idx)],
                    choices=[str(c) for c in choices],
                    correct_choice_idx=idx,
                    metric_hint="multiple_choice",
                    metadata={
                        "pid": row.get("pid"),
                        "answer_type": row.get("answer_type"),
                        "question_type": "multi_choice",
                    },
                )
                continue
            # MC avec answer non trouvée dans choices → fallback VQA ouvert (rare)

        # Free-form (text / int / float) ou MC fallback
        yield UnifiedSample(
            id=_sid("mathvista", i, str(row.get("pid") or "")),
            source="mathvista",
            task="vqa",
            image=_ensure_pil(img),
            question=question,
            answers=[answer],
            metric_hint="vqa_accuracy",
            metadata={
                "pid": row.get("pid"),
                "answer_type": row.get("answer_type"),
                "question_type": question_type or "free_form",
                "unit": row.get("unit"),
                "precision": row.get("precision"),
            },
        )


# ---- Registre ---------------------------------------------------------------

LOADERS: dict[str, Callable[..., Iterator[UnifiedSample]]] = {
    "docvqa": load_docvqa,
    "textvqa": load_textvqa,
    "mmmu": load_mmmu,
    "vqav2": load_vqav2,
    "chartqa": load_chartqa,
    "ai2d": load_ai2d,
    "realworldqa": load_realworldqa,
    "coco_captions": load_coco_captions,
    "ok_vqa": load_okvqa,
    # v0.3
    "scienceqa": load_scienceqa,
    "infographicvqa": load_infographicvqa,
    "mathvista": load_mathvista,
}


def load_source(
    name: str,
    split: str,
    max_samples: int | None = None,
) -> Iterator[UnifiedSample]:
    """Point d'entrée unique — dispatch par nom.

    Le `split` est l'étiquette canonique ('train'/'val'/'test'), pas le nom
    natif du dataset. La traduction est faite via DatasetSpec.splits.

    `max_samples` : si set, active le streaming + s'arrête après N samples
    (utile pour sandbox / CI ; ne pas passer en prod Runpod si on veut full).
    """
    if name not in LOADERS:
        raise KeyError(f"Dataset inconnu : {name}. Choix : {list(LOADERS)}")
    spec = ALL_DATASETS[name]
    native = spec.splits.get(split)
    if native is None:
        # pas de split natif — pas d'yield, le mix se débrouillera
        return iter(())
    return LOADERS[name](native, max_samples=max_samples)
