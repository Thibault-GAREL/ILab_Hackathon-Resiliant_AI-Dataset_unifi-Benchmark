"""Backend vLLM pour les modèles HF compatibles."""

from __future__ import annotations

import re
import time
from inspect import signature
from typing import Any, Callable

GenerateFn = Callable[..., str]


def _post_process_pred(text: str, task: str) -> str:
    text = re.sub(r"\*+", "", str(text)).strip()
    text = text.split("\n", 1)[0].strip()
    if task == "multiple_choice":
        match = re.search(r"\b([A-H])\b", text.upper())
        if match:
            return match.group(1)
    return text


def _build_prompt(prompt: str | None, sample: dict[str, Any]) -> str:
    task = sample.get("task") or "vqa"
    choices = list(sample.get("choices") or [])
    question = str(prompt or sample.get("question") or "").strip()

    if task == "multiple_choice" and choices:
        labeled = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
        return f"{question}\n\n{labeled}\n\nAnswer with ONLY the letter of the correct choice."
    if task == "captioning":
        return "Describe this image in one short sentence. No explanation, no markdown."
    return f"{question}\n\nAnswer in as few words as possible. No explanation, no markdown."


def make_vllm_generate_fn(
    model_path: str,
    max_new_tokens: int = 32,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 1,
    dtype: str = "auto",
    quantization: str | None = None,
    kv_cache_dtype: str | None = None,
    max_soft_tokens: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    enforce_eager: bool = False,
    revision: str | None = None,
    seed: int | None = 42,
) -> GenerateFn:
    try:
        import vllm
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError('vLLM non installé. Sur RunPod : pip install "vllm==0.19.1" avant --backend vllm.') from exc

    processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision:
        processor_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "limit_mm_per_prompt": {"image": 1},
        "enforce_eager": enforce_eager,
        "trust_remote_code": True,
    }
    if dtype != "auto":
        llm_kwargs["dtype"] = dtype
    if quantization:
        llm_kwargs["quantization"] = quantization
    if kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if revision:
        llm_kwargs["revision"] = revision

    print(f"[load vllm] {model_path}")
    print(f"[vllm kwargs] {llm_kwargs}")
    load_t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    print(f"[load vllm] ok en {time.perf_counter() - load_t0:.1f}s (vllm={getattr(vllm, '__version__', 'unknown')})")
    sampling_fields = set(signature(SamplingParams).parameters)

    class VLLMGenerateFn:
        def __init__(self) -> None:
            self.llm = llm
            self.processor = processor
            self.closed = False

        def close(self) -> None:
            if self.closed:
                return
            self.closed = True
            for obj in (self.llm, getattr(self.llm, "llm_engine", None)):
                shutdown = getattr(obj, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                        break
                    except Exception:
                        pass
            self.llm = None
            self.processor = None

        def __call__(self, image, prompt, sample=None, **kwargs) -> str:
            sample = sample or {}
            task = sample.get("task", "vqa")
            sampling_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_new_tokens,
            }
            if seed is not None and "seed" in sampling_fields:
                sampling_kwargs["seed"] = seed
            sampling = SamplingParams(**sampling_kwargs)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _build_prompt(prompt, sample)},
                    ],
                }
            ]
            request: dict[str, Any] = {
                "prompt": self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "multi_modal_data": {"image": image.convert("RGB")},
            }
            if max_soft_tokens:
                request["mm_processor_kwargs"] = {"max_soft_tokens": max_soft_tokens}
            outputs = self.llm.generate(request, sampling_params=sampling)
            text = outputs[0].outputs[0].text if outputs else ""
            return _post_process_pred(text, str(task))

    return VLLMGenerateFn()
