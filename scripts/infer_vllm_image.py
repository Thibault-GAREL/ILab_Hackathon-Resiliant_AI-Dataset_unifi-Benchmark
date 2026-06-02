#!/usr/bin/env python3
"""Test rapide d'une image avec un label vLLM du YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "models.yaml"

sys.path.insert(0, str(ROOT))


def _load_config() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML est requis. Installer requirements.txt ou lancer dans le container benchmark."
        ) from exc
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _model_from_label(cfg: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [model for model in cfg.get("models", []) if model.get("label") == label]
    if not matches:
        labels = ", ".join(
            model.get("label", "?")
            for model in cfg.get("models", [])
            if model.get("backend") == "vllm"
        )
        raise SystemExit(
            f"Label vLLM inconnu {label!r}. Labels vLLM disponibles: {labels}"
        )
    if len(matches) > 1:
        raise SystemExit(f"Label {label!r} duplique dans {CONFIG_PATH}.")
    model = dict(matches[0])
    if model.get("backend") != "vllm":
        raise SystemExit(
            f"Label {label!r} utilise backend {model.get('backend', 'transformers')!r}, pas vllm."
        )
    return model


def _pick(args_value: Any, model: dict[str, Any], defaults: dict[str, Any], key: str) -> Any:
    if args_value is not None:
        return args_value
    if key in model:
        return model[key]
    return defaults.get(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference image directe via le backend vLLM du repo."
    )
    parser.add_argument("image", type=Path, help="Chemin vers une image locale.")
    parser.add_argument(
        "--label",
        default="gemma4-e4b-vllm-fp8-kvfp8-soft140-t16",
        help="Label vLLM dans configs/models.yaml.",
    )
    parser.add_argument("--model-path", help="Override model_path du label.")
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="Question ou prompt pour l'image.",
    )
    parser.add_argument(
        "--task",
        default="vqa",
        choices=["vqa", "multiple_choice", "captioning"],
        help="Mode de formatting du prompt cote backend benchmark.",
    )
    parser.add_argument(
        "--choice",
        action="append",
        dest="choices",
        help="Option QCM. Repeter le flag pour A/B/C/...",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--max-soft-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image introuvable: {args.image}")

    cfg = _load_config()
    defaults = cfg.get("defaults", {})
    model = _model_from_label(cfg, args.label)
    model_path = args.model_path or model.get("model_path")
    if not model_path:
        raise SystemExit("--model-path est requis si le label n'a pas de model_path.")

    from PIL import Image
    from src.models.vllm_backend import make_vllm_generate_fn

    gen = make_vllm_generate_fn(
        model_path=model_path,
        max_new_tokens=int(
            _pick(args.max_new_tokens, model, defaults, "max_new_tokens") or 32
        ),
        tensor_parallel_size=int(
            _pick(args.tensor_parallel_size, model, defaults, "vllm_tensor_parallel_size")
            or 1
        ),
        max_model_len=int(
            _pick(args.max_model_len, model, defaults, "vllm_max_model_len") or 2048
        ),
        gpu_memory_utilization=float(
            _pick(
                args.gpu_memory_utilization,
                model,
                defaults,
                "vllm_gpu_memory_utilization",
            )
            or 0.9
        ),
        max_num_seqs=int(
            _pick(args.max_num_seqs, model, defaults, "vllm_max_num_seqs") or 1
        ),
        dtype=str(_pick(args.dtype, model, defaults, "vllm_dtype") or "auto"),
        quantization=_pick(args.quantization, model, defaults, "vllm_quantization"),
        kv_cache_dtype=_pick(args.kv_cache_dtype, model, defaults, "vllm_kv_cache_dtype"),
        max_soft_tokens=_pick(args.max_soft_tokens, model, defaults, "vllm_max_soft_tokens"),
        temperature=float(_pick(args.temperature, model, defaults, "temperature") or 1.0),
        top_p=float(_pick(args.top_p, model, defaults, "top_p") or 0.95),
        top_k=int(_pick(args.top_k, model, defaults, "top_k") or 64),
        enforce_eager=bool(
            args.enforce_eager
            or model.get("vllm_enforce_eager", defaults.get("vllm_enforce_eager", False))
        ),
        revision=args.revision or model.get("revision"),
        seed=_pick(args.seed, model, defaults, "vllm_seed"),
    )

    sample: dict[str, Any] = {"task": args.task}
    if args.choices:
        sample["choices"] = args.choices

    try:
        with Image.open(args.image) as image:
            pred = gen(image=image, prompt=args.prompt, sample=sample)
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()

    print(pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
