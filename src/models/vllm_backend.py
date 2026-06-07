"""Backend vLLM pour les modèles HF compatibles."""

from __future__ import annotations

import atexit
import re
import threading
import time
from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable

GenerateFn = Callable[..., str]


@dataclass
class _AdaptiveSoftTokenState:
    default_soft_tokens: int
    throttled_soft_tokens: int
    peak_power_w: float | None
    peak_active_power_w: float | None
    baseline_power_w: float
    cooldown_requests: int
    throttle_remaining: int = 0
    trigger_count: int = 0
    max_power_w: float | None = None

    def next_soft_tokens(self, requested: int | None) -> int | None:
        if requested is None:
            requested = self.default_soft_tokens
        if self.throttle_remaining <= 0:
            return requested
        self.throttle_remaining -= 1
        return min(requested, self.throttled_soft_tokens)

    def observe(self, power_max_w: float | None) -> None:
        if power_max_w is None:
            return
        self.max_power_w = power_max_w if self.max_power_w is None else max(self.max_power_w, power_max_w)
        active_power_w = power_max_w - self.baseline_power_w
        triggered = False
        if self.peak_power_w is not None and power_max_w >= self.peak_power_w:
            triggered = True
        if self.peak_active_power_w is not None and active_power_w >= self.peak_active_power_w:
            triggered = True
        if triggered and self.cooldown_requests > 0:
            self.trigger_count += 1
            self.throttle_remaining = max(self.throttle_remaining, self.cooldown_requests)

    def runtime_stats(self) -> dict[str, Any]:
        return {
            "default_soft_tokens": self.default_soft_tokens,
            "throttled_soft_tokens": self.throttled_soft_tokens,
            "peak_power_w": self.peak_power_w,
            "peak_active_power_w": self.peak_active_power_w,
            "baseline_power_w": self.baseline_power_w,
            "cooldown_requests": self.cooldown_requests,
            "throttle_remaining": self.throttle_remaining,
            "trigger_count": self.trigger_count,
            "max_power_w": self.max_power_w,
        }


class _NvmlPowerMonitor:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.2) -> None:
        self.gpu_index = gpu_index
        self.interval_s = max(interval_s, 0.05)
        self.error: str | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._powers: list[float] = []

    def start(self) -> _NvmlPowerMonitor:
        try:
            import pynvml as pynvml_mod

            self._pynvml = pynvml_mod
            self._pynvml.nvmlInit()
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as exc:
            self.error = repr(exc)
            return self
        self._thread = threading.Thread(target=self._run, name="vllm-nvml-soft-token-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def mark(self) -> int:
        with self._lock:
            return len(self._powers)

    def max_since(self, mark: int) -> float | None:
        with self._lock:
            window = self._powers[mark:]
        return max(window) if window else None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                power_w = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                with self._lock:
                    self._powers.append(power_w)
            except Exception as exc:
                self.error = repr(exc)
                return
            self._stop.wait(self.interval_s)


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
    adaptive_soft_tokens: str = "none",
    adaptive_throttled_soft_tokens: int = 70,
    adaptive_peak_power_w: float | None = None,
    adaptive_peak_active_power_w: float | None = None,
    adaptive_baseline_power_w: float = 0.0,
    adaptive_cooldown_requests: int = 3,
    adaptive_gpu_index: int = 0,
    adaptive_sample_interval_s: float = 0.2,
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
    adaptive_state: _AdaptiveSoftTokenState | None = None
    power_monitor: _NvmlPowerMonitor | None = None
    if adaptive_soft_tokens != "none":
        adaptive_state = _AdaptiveSoftTokenState(
            default_soft_tokens=max_soft_tokens or 140,
            throttled_soft_tokens=adaptive_throttled_soft_tokens,
            peak_power_w=adaptive_peak_power_w,
            peak_active_power_w=adaptive_peak_active_power_w,
            baseline_power_w=adaptive_baseline_power_w,
            cooldown_requests=adaptive_cooldown_requests,
        )
        power_monitor = _NvmlPowerMonitor(
            gpu_index=adaptive_gpu_index,
            interval_s=adaptive_sample_interval_s,
        ).start()
        atexit.register(power_monitor.stop)
        print(
            "[adaptive soft tokens] "
            f"policy={adaptive_soft_tokens} throttle={adaptive_throttled_soft_tokens} "
            f"peak_power={adaptive_peak_power_w} peak_active={adaptive_peak_active_power_w} "
            f"nvml_error={power_monitor.error}"
        )

    class VLLMGenerateFn:
        def __init__(self) -> None:
            self.llm = llm
            self.processor = processor
            self.adaptive_policy = adaptive_soft_tokens
            self.adaptive_state = adaptive_state
            self.power_monitor = power_monitor
            self.closed = False

        def runtime_stats(self) -> dict[str, Any]:
            adaptive_stats: dict[str, Any] = {
                "policy": self.adaptive_policy,
                "enabled": self.adaptive_state is not None,
            }
            if self.adaptive_state is not None:
                adaptive_stats.update(self.adaptive_state.runtime_stats())
            if self.power_monitor is not None:
                adaptive_stats["nvml_error"] = self.power_monitor.error
            return {"adaptive_soft_tokens": adaptive_stats}

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
            if power_monitor is not None:
                power_monitor.stop()
            self.llm = None
            self.processor = None

        def __call__(self, image, prompt, sample=None, **kwargs) -> str:
            sample = sample or {}
            task = sample.get("task", "vqa")
            effective_soft_tokens = (
                adaptive_state.next_soft_tokens(max_soft_tokens)
                if adaptive_state is not None
                else max_soft_tokens
            )
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
            if effective_soft_tokens:
                request["mm_processor_kwargs"] = {"max_soft_tokens": effective_soft_tokens}
            mark = power_monitor.mark() if power_monitor is not None else None
            outputs = self.llm.generate(request, sampling_params=sampling)
            if adaptive_state is not None and power_monitor is not None:
                adaptive_state.observe(power_monitor.max_since(mark or 0))
            text = outputs[0].outputs[0].text if outputs else ""
            return _post_process_pred(text, str(task))

    return VLLMGenerateFn()
