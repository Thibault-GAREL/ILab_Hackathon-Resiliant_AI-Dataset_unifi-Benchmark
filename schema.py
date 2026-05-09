"""
Schéma unifié pour les 9 datasets du Resilient AI Challenge.

Un sample unifié = une interaction image → texte cohérente, quelle que soit
la nature du dataset source (VQA / MC / captioning).

Choix de design :
- l'image est stockée en **bytes JPEG compressés** (pas en PIL.Image) pour
  plafonner la RAM pendant le build — une image PIL décompressée fait
  typiquement 10-20× la taille de son équivalent JPEG. HuggingFace `datasets`
  accepte le format {'bytes': ..., 'path': None} dans sa feature Image() ;
- `answers` est toujours une liste (VQAv2 a 10 réponses humaines, les autres
  en ont 1 — on homogénéise) ;
- `choices` + `correct_choice_idx` ne sont peuplés que pour MMMU / AI2D ;
- `metadata` conserve tout champ original utile (bbox, question_type,
  image_id…) pour le diagnostic post-benchmark sans peser sur le mainline.

Format `messages` est produit à la demande par `to_chat()` pour éviter de
stocker 9× le même template et préserver la taille du dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from io import BytesIO
from typing import Any, Literal, Optional
from PIL import Image


TaskType = Literal["vqa", "captioning", "multiple_choice"]


@dataclass
class UnifiedSample:
    """Un sample après harmonisation.

    Représentation canonique utilisée par :
    - le mix sampler (mix.py) qui ne touche qu'à `source` et aux index,
    - le benchmark (benchmark.py) qui dispatche sur `task` + `metric_hint`,
    - le pré-processeur de distillation qui consomme `to_chat()`.
    """
    id: str
    source: str                      # nom court du dataset (clé de ALL_DATASETS)
    task: TaskType
    image: bytes | None              # JPEG bytes (None si tâche text-only, jamais dans notre cas)
    question: str                    # prompt utilisateur
    answers: list[str]               # liste non-vide ; ordre : [réponse canonique, ... variantes]
    choices: list[str] | None = None # MC uniquement
    correct_choice_idx: int | None = None
    metric_hint: str = "vqa_accuracy"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- Validation minimale -------------------------------------------------
    def __post_init__(self) -> None:
        if not self.answers:
            raise ValueError(f"Sample {self.id} ({self.source}) a zéro réponse.")
        if self.task == "multiple_choice":
            if self.choices is None or self.correct_choice_idx is None:
                raise ValueError(
                    f"Sample MC {self.id} ({self.source}) sans choices/idx."
                )
            if not (0 <= self.correct_choice_idx < len(self.choices)):
                raise ValueError(
                    f"Sample MC {self.id} : idx {self.correct_choice_idx} "
                    f"hors des {len(self.choices)} choix."
                )

    # ---- Sérialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Pour HuggingFace `datasets`.

        L'image est passée au format {'bytes': ..., 'path': None} que la
        feature `Image()` de `datasets` sait consommer directement (évite
        la re-décompression PIL qu'un cast depuis Image.Image provoquerait).
        """
        d = asdict(self)
        # asdict laisse les bytes intacts, on les enveloppe pour HF Image()
        d["image"] = {"bytes": self.image, "path": None} if self.image else None
        return d

    # ---- Accès PIL à la demande (benchmark / training) -----------------------
    def to_pil(self) -> "Image.Image | None":
        """Décompresse les bytes en PIL.Image à la demande.

        À utiliser uniquement au moment de l'inférence / du preprocessing modèle
        — pas pendant le build (où on veut rester en bytes pour économiser RAM).
        """
        if self.image is None:
            return None
        return Image.open(BytesIO(self.image)).convert("RGB")

    # ---- Format conversation pour Gemma --------------------------------------
    def to_chat(self, include_answer: bool = True) -> list[dict[str, Any]]:
        """Formate en messages Gemma-compatible.

        Training : include_answer=True (le target est dans assistant).
        Inference/benchmark : include_answer=False (on génère la réponse).

        Format suit le style multi-turn instruction-tuning de Gemma 3n, qui
        accepte content soit str soit liste de parts {type, text/image}.
        """
        user_parts: list[dict[str, Any]] = [{"type": "image"}]

        if self.task == "multiple_choice" and self.choices:
            # Numérotation A/B/C/D conventionnelle pour les MC
            labeled = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(self.choices)
            )
            text = (
                f"{self.question.strip()}\n\n{labeled}\n\n"
                f"Answer with the letter of the correct choice."
            )
        elif self.task == "captioning":
            text = self.question.strip() or "Describe this image."
        else:  # vqa
            text = self.question.strip()

        user_parts.append({"type": "text", "text": text})
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_parts}]

        if include_answer:
            if self.task == "multiple_choice" and self.correct_choice_idx is not None:
                target = chr(65 + self.correct_choice_idx)
            else:
                target = self.answers[0]
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": target}]}
            )

        return messages


# ---- Schema pour HuggingFace datasets ---------------------------------------

def hf_features() -> dict[str, Any]:
    """Schema `Features` pour `datasets.Dataset.from_list(...).cast(...)`.

    À appeler après import de `datasets` côté build_dataset.py (évite un
    import lourd au chargement de ce module).
    """
    from datasets import Features, Value, Image as HFImage, Sequence

    return Features({
        "id": Value("string"),
        "source": Value("string"),
        "task": Value("string"),
        "image": HFImage(),
        "question": Value("string"),
        "answers": Sequence(Value("string")),
        "choices": Sequence(Value("string")),
        "correct_choice_idx": Value("int32"),
        "metric_hint": Value("string"),
        # metadata : on le sérialise en JSON string pour rester schema-stable
        "metadata": Value("string"),
    })
