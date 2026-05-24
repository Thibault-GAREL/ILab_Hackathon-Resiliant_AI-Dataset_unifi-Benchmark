# 🔬 Resilient AI — Unified Dataset + VLM Benchmark

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.44%2B-yellow.svg)
![llama.cpp](https://img.shields.io/badge/llama--cpp--python-CUDA%2012.1-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)
![CodeCarbon](https://img.shields.io/badge/CodeCarbon-2.4-green.svg)
![Docker](https://img.shields.io/badge/Docker-GHCR-2496ED.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="https://github.com/Thibault-GAREL/Thibault-GAREL/raw/main/Logo_Group_Projects/group_resilient_sq.png" alt="Resilient AI Challenge Logo" width="200">
</p>

---

## 📝 Project Description

This project is a **unified benchmark pipeline** for evaluating multimodal large language models (VLMs) on the **PEReN Resilient AI Challenge** at ECE-ILAB. It merges **12 vision-language datasets** (DocVQA, VQAv2, ChartQA, COCO Captions, MMMU, AI2D, etc.) into a single HuggingFace repo, then evaluates any **Gemma 4 / Gemma 3n** checkpoint with per-source metrics + **energy tracking via CodeCarbon**.

I built this to understand how **VLM compression** (AWQ, GGUF, FP8) trades quality for energy on real benchmarks — the PEReN ranking is **energy-dominated** once you cross the 80% accuracy threshold, so the cheapest model that stays above the bar wins. No magic tricks, just measure everything: kWh, J/sample, gCO2eq, and a 32-column registry that grows by 1 line per run.

The whole stack runs in a **Docker image auto-built by GitHub Actions** (GHCR) and deployable on Runpod in 1 click. Plenty of fun debugging CUDA libs along the way 😆.

---

## ⚙️ Features

  📦 **Unified dataset** — 12 vision-language sources merged into 1 HuggingFace repo (`ECE-ILAB/resilient-ai-unified`, 423k samples, sqrt-balanced)

  🎯 **Per-source metrics** — **ANLS** for DocVQA, **VQA accuracy** for VQAv2, **CIDEr** for COCO Captions, **Multiple Choice** for AI2D/MMMU, **Relaxed accuracy** for ChartQA

  ⚡ **Energy tracking** — **CodeCarbon** per-source + aggregated KPIs (`J/sample`, `samples/J`, `kWh`, `gCO2eq`)

  📊 **Append-only registry** — every run adds 1 line to `results/registry.csv` (32 columns) for comparing 50+ runs without losing history

  🔁 **Multi-backend** — **transformers** (BF16, AWQ) + **llama-cpp-python** (GGUF multimodal with mmproj)

  🐳 **Docker auto-suffisant** — torch pinned + llama-cpp-python CUDA wheel + autoawq + compressed-tensors + ldconfig nvidia — zero pip install at runtime

  📐 **Deterministic subsample** — 10% per-source via `seed XOR hash(source)`, 100% for final submission run

  🚀 **Runpod-ready** — wrapper `bench_safe.sh` auto-configures `HF_HOME`, `LD_LIBRARY_PATH`, symlinks `/workspace`, cleans HF cache between runs to stay within volume quota

---

## Example Outputs

Sample registry comparing Gemma 4 variants on **RTX 4090 / Runpod EU-CZ-1**, `limit_per_source=20` (240 samples total):

| Model | macro_avg | J/sample | samples/J | Notes |
|---|---|---|---|---|
| `gemma4-e4b-bf16` | 0.3996 | 117.5 | 0.0085 | Baseline reference |
| `gemma4-e2b-bf16` | 0.3569 | 114.2 | 0.0088 | Lower bound (2× smaller) |
| `gemma4-e4b-awq-int4` | 0.4080 | 135.0 | 0.0074 | AWQ INT4 (cyankiwi) |
| `gemma4-e4b-awq-int8` | 0.3958 | 125.0 | 0.0080 | AWQ INT8 (cyankiwi) |
| `gemma4-e4b-gguf-q8` | ~0.40* | ~110* | ~0.0090* | GGUF Q8_0 (quasi-lossless) |
| `gemma4-e4b-gguf-q4km` | ~0.36* | ~95* | ~0.0105* | GGUF Q4_K_M (4-bit community reference) |

*\*GGUF scores depend on llama-cpp-python multimodal support for Gemma 4 — currently experimental.*

### 📝 Notes & Observations

- **AWQ INT4** is competitive with BF16 (sometimes higher on 240 samples — variance) — confirms compression is viable
- **AWQ consumes more energy than BF16** on short runs (de-quant overhead at init not amortized)
- **bnb-4bit removed** : bug FP4 in `bitsandbytes` for Gemma 4 multimodal (`AssertionError: weight.shape[1] == 1`) — not patched upstream
- **The PEReN KPI is `samples_per_joule`** — sort the registry by this column descending to find the best efficiency

---

## ⚙️ How it works

  📂 The pipeline loads a **DatasetDict** with 4 splits (train / validation / test / calibration), groups samples **by source**

  🎲 Subsamples deterministically via `seed XOR hash(source)` — same seed = same samples → comparable runs across models

  🤖 Loads the model via **AutoModelForImageTextToText** (transformers) or **Llama** (llama-cpp-python) with multimodal `mmproj`

  💬 Generates with a **task-aware prompt** : MC = *"Answer ONLY the letter"*, VQA = *"as few words as possible"*, captioning = *"one short sentence"*

  ✂️ Post-processes the output : strips markdown bold/italic, keeps the first line, extracts a letter A–E for MC questions

  📏 Scores per-source with the matching metric, aggregates `micro_avg` (sample-weighted) + `macro_avg` (source-equal)

  🔋 Wraps each source in an **EnergyTracker** (CodeCarbon) → kWh + CO2 + J/sample → `score_per_kwh` KPI for PEReN

  📝 Appends 1 line to `results/registry.csv` + dumps detailed JSON + per-source energy CSV + 1-page summary text

---

## 🗺️ Architecture Diagram

The project is a **multi-backend benchmarking pipeline** rather than a trained model. End-to-end flow:

```
┌──────────────────┐    ┌──────────────────┐    ┌───────────────────────┐
│  HuggingFace Hub │    │  Dataset local   │    │  Per-source loop      │
│  ECE-ILAB/...    │ →  │  /workspace/...  │ →  │  ├─ subsample 10%     │
│  (12 sources)    │    │  (DatasetDict)   │    │  ├─ EnergyTracker on  │
└──────────────────┘    └──────────────────┘    │  ├─ for each sample:  │
                                                 │  │   generate_fn()    │
                                                 │  └─ score_sample()    │
                                                 └─────────┬─────────────┘
                                                           │
                                       ┌───────────────────┼─────────────────────┐
                                       ▼                   ▼                     ▼
                            ┌──────────────────┐ ┌──────────────────┐ ┌────────────────────┐
                            │   transformers   │ │   llama_cpp      │ │   Metrics dispatch │
                            │   AutoModelFor   │ │   Llama +        │ │   ANLS / VQA acc / │
                            │   ImageTextTo    │ │   mmproj (GGUF)  │ │   CIDEr / MC /     │
                            │   Text (BF16/AWQ)│ │   multimodal     │ │   Relaxed accuracy │
                            └──────────────────┘ └──────────────────┘ └────────────────────┘
                                                           │
                                                           ▼
                                                ┌────────────────────────┐
                                                │ results/registry.csv   │
                                                │   (append-only, 32 cols)│
                                                │ results/runs/*.json    │
                                                │ results/runs/*_energy. │
                                                │ reports/<label>.json   │
                                                └────────────────────────┘
```

**Key components:**

  📚 `src/data/` → 12 loaders + `UnifiedSample` schema + sqrt-balanced mix strategy

  🧪 `src/validation/benchmark.py` → `evaluate_model()` with per-source dispatch + Gemma-aware prompt formatting

  🔋 `src/tracking/` → CodeCarbon wrapper + `RegistryEntry` (32-column schema)

  🦙 `src/models/llama_cpp_backend.py` → GGUF multimodal inference with HF glob support for mmproj

  🐳 `Dockerfile` → torch 2.5.1 pinned + llama-cpp-python CUDA 12.1 wheel + ldconfig nvidia + ENV `HF_HOME=/workspace/hf-cache`

---

## 📂 Repository structure

```bash
├── configs/
│   └── models.yaml             # 21 models: transformers (BF16+AWQ) + GGUF (bartowski + axelbrons)
│
├── src/                        # Project code (importable from scripts/)
│   ├── config.py               # DatasetSpec, ALL_DATASETS, MixConfig
│   ├── data/                   # 12 loaders + UnifiedSample + mix sqrt/uniform/temperature
│   ├── models/                 # llama_cpp_backend (multimodal GGUF)
│   ├── validation/             # benchmark.py + metrics.py + eval_ood.py (MMStar/MME/POPE)
│   └── tracking/               # energy_tracker (CodeCarbon) + run_registry
│
├── scripts/                    # CLI entrypoints
│   ├── build_dataset.py        # Build + push the unified dataset
│   ├── eval_loop.py            # Benchmark a single model
│   ├── bench_all.py            # Batch orchestrator (reads configs/models.yaml)
│   ├── bench_safe.sh           # Self-contained Runpod wrapper (HF_HOME, ldconfig, symlinks)
│   ├── entrypoint.sh           # Docker ENTRYPOINT (mkdir + symlinks at boot)
│   ├── runpod_setup.sh         # New-pod setup (one-shot)
│   └── runpod_run.sh           # Build the dataset on Runpod
│
├── results/
│   ├── registry.csv            # APPEND-ONLY, 1 row / run, 32 columns
│   └── runs/                   # Detailed JSON + per-source energy CSV
│
├── reports/                    # eval_loop JSON reports (1 per run)
│
├── .github/workflows/
│   └── docker-publish.yml      # Build + push image to GHCR on every commit
│
├── Dockerfile                  # Self-contained image (torch + cuda + llama-cpp-python)
├── docker-compose.yml          # Local run
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 💻 Run it on Your PC

⚠️ You need a **CUDA-compatible GPU** with ≥6 GB VRAM. For BF16 baselines, prefer ≥24 GB (RTX 4090). For GGUF Q4/Q5 quantizations, 8 GB suffit (GTX 1660 Ti OK).

Clone the repository and install dependencies:

```bash
git clone https://github.com/Thibault-GAREL/ILab_Hackathon-Resiliant_AI-Dataset_unifi-Benchmark.git
cd ILab_Hackathon-Resiliant_AI-Dataset_unifi-Benchmark

python -m venv .venv # if you don't have a virtual environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
huggingface-cli login
```

### Benchmark a single model

```bash
python scripts/eval_loop.py \
    --mode real \
    --model_path google/gemma-4-E4B-it \
    --dataset_hub ECE-ILAB/resilient-ai-unified \
    --subsample_pct 100 \
    --limit_per_source 20 \
    --country_iso FRA \
    --run_label gemma4-e4b-bf16-local \
    --out reports/gemma4-e4b-bf16-local.json
```

### Run the full batch via Docker (recommended)

```bash
# Pull the prebuilt image (~10 GB, includes torch + cuda + llama-cpp + autoawq)
docker pull ghcr.io/thibault-garel/ilab_hackathon-resiliant_ai-dataset_unifi-benchmark:latest

# Run with GPU
docker compose run --rm bench bash scripts/bench_safe.sh
```

### Build the unified dataset from scratch (one-shot, ~40 GB)

```bash
export HF_TOKEN=hf_xxxxx
bash scripts/runpod_run.sh
```

---

## 🚀 Runpod — Battle-tested recipes (copy-paste ready)

All the blocks below were tested end-to-end on **RTX 4090 / Runpod EU-CZ-1** and run reproducibly.

### A. Sync the code (after each new pod boot)

```bash
rm -rf /tmp/repo
git clone --depth 1 https://github.com/Thibault-GAREL/ILab_Hackathon-Resiliant_AI-Dataset_unifi-Benchmark.git /tmp/repo
cp /tmp/repo/configs/models.yaml /app/configs/models.yaml
cp /tmp/repo/src/models/llama_cpp_backend.py /app/src/models/llama_cpp_backend.py
cp /tmp/repo/scripts/bench_all.py /app/scripts/bench_all.py
cp /tmp/repo/scripts/bench_safe.sh /app/scripts/bench_safe.sh
chmod +x /app/scripts/bench_safe.sh

python /app/scripts/bench_all.py --list
```

### B. Run a subset of models (~60 min on RTX 4090)

```bash
nohup python /app/scripts/bench_all.py \
    --label gemma4-e4b-bf16 --label gemma4-e2b-bf16 \
    --label gemma4-e4b-axelbrons-q2k --label gemma4-e4b-axelbrons-q3ks \
    --label gemma4-e4b-axelbrons-q3km --label gemma4-e4b-axelbrons-q3kl \
    --label gemma4-e4b-axelbrons-q4ks --label gemma4-e4b-axelbrons-q4km \
    --label gemma4-e4b-axelbrons-q5ks --label gemma4-e4b-axelbrons-q5km \
    --label gemma4-e4b-axelbrons-q6k --label gemma4-e4b-axelbrons-q8 \
    --dataset_local /workspace/code/data/unified \
    --limit_per_source 20 --cleanup_cache_after --skip_existing \
    > /workspace/bench.log 2>&1 & disown

tail -f /workspace/bench.log
```

### C. Re-test models already in the registry

```bash
# Remove the matching reports/JSON (otherwise --skip_existing skips them)
rm -f /workspace/reports/gemma4-e4b-bf16.json /workspace/reports/gemma4-e2b-bf16.json

# (Optional) filter out the matching rows from the registry
python <<'PY'
import csv, shutil
src = '/workspace/results/registry.csv'
shutil.copy(src, src + '.bak')
with open(src) as f:
    rows = list(csv.DictReader(f)); fn = list(rows[0].keys())
labels_to_remove = {'gemma4-e4b-bf16', 'gemma4-e2b-bf16'}
keep = [r for r in rows if r['run_label'] not in labels_to_remove]
print(f'{len(rows)} -> {len(keep)} runs')
with open(src, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
    for r in keep: w.writerow(r)
PY
```

### D. Read the registry on the pod (sorted by energy efficiency)

```bash
python -c "
import csv
with open('/workspace/results/registry.csv') as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r.get('samples_per_joule', 0) or 0), reverse=True)
print(f'═══ {len(rows)} runs (sorted by samples/J desc) ═══')
print(f'{\"label\":<40} {\"macro\":>8} {\"J/sample\":>10} {\"samples/J\":>12}')
print('-' * 78)
for r in rows:
    print(f'{r[\"run_label\"]:<40} {r[\"macro_avg\"]:>8} {r[\"joules_per_sample\"]:>10} {r[\"samples_per_joule\"]:>12}')
"
```

### E. Pull the registry back to your PC

```bash
# On the pod
rm -f /workspace/results.zip 2>/dev/null
runpodctl send /workspace/results/registry.csv
# → copy the code shown
```

```powershell
# On your Windows PC (PowerShell), from the project folder
cd "d:\Loisir\Code_python\ILab_Resilient_AI_-_Dataset_unifié+Benchmark"

if (!(Test-Path .\runpodctl.exe)) {
    Invoke-WebRequest -Uri "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe" -OutFile "runpodctl.exe"
}
.\runpodctl.exe receive <code-from-pod>
```

### F. Test a single model (one-off debug)

```bash
python /app/scripts/bench_all.py \
    --label gemma4-e4b-gguf-q4km \
    --dataset_local /workspace/code/data/unified \
    --limit_per_source 5
```

⚠️ **Always pass `--dataset_local`** otherwise `bench_all.py` falls back to the `dataset_hub` from the YAML and tries to re-download ~50 GB from HuggingFace.

### G. Disk cleanup on `/workspace` (quota exceeded)

```bash
df -h /workspace
du -sh /workspace/hf-cache/hub/* 2>/dev/null | sort -h

# Drop already-benchmarked models (the registry rows still persist on disk)
rm -rf /workspace/hf-cache/hub/models--google--gemma-4-E4B-it
rm -rf /workspace/hf-cache/hub/models--google--gemma-4-E2B-it
rm -rf /workspace/hf-cache/hub/models--cyankiwi--*
rm -rf /workspace/hf-cache/hub/datasets--*
rm -rf /workspace/hf-cache/xet
rm -rf /workspace/pip-cache /workspace/tmp/*

df -h /workspace
```

---

## 📊 Key `eval_loop.py` flags

| Flag | Default | Purpose |
|---|---|---|
| `--mode` | `real` | `real` / `oracle` / `constant` (debug modes) |
| `--backend` | `transformers` | `transformers` / `llama_cpp` |
| `--model_path` | – | required when `backend=transformers` |
| `--gguf_repo` / `--gguf_filename` | – | required when `backend=llama_cpp` (glob accepted) |
| `--mmproj_repo` / `--mmproj_filename` | – | multimodal projector for GGUF VLM |
| `--dataset_local` / `--dataset_hub` | – | one of the two is required |
| `--subsample_pct` | `10.0` | % per-source to evaluate (100 = full split) |
| `--limit_per_source` | – | hard cap post-subsample (smoke test) |
| `--country_iso` | `FRA` | CodeCarbon country code (CZE for Runpod EU-CZ-1) |
| `--run_label` | derived | short identifier in the registry |
| `--out` | required | main JSON report path |

---

## 📖 Inspiration / Sources

This project is part of the **ECE-ILAB Resilient AI Challenge** (PEReN-evaluated), built on top of:

- 🤗 [HuggingFace `datasets`](https://huggingface.co/docs/datasets) for unified data loading
- 🤖 [HuggingFace `transformers`](https://huggingface.co/docs/transformers) (Gemma 4 + Gemma 3n multimodal)
- 🦙 [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) for GGUF inference with CUDA 12.1 wheel
- 🔋 [`CodeCarbon`](https://github.com/mlco2/codecarbon) for energy tracking (PEReN KPI)
- 📦 GGUF quantizations from [`bartowski`](https://huggingface.co/bartowski) and [`axelbrons`](https://huggingface.co/axelbrons)
- 🗜️ AWQ checkpoints from [`cyankiwi`](https://huggingface.co/cyankiwi)

I used **Claude AI** to scaffold the multi-backend dispatching (transformers ↔ llama_cpp ↔ vLLM), debug the CUDA libs resolution (libcuda.so.1 vs libcudart.so.12), and accelerate the Dockerfile setup with `ldconfig` nvidia paths.

**Contributors:**
- 🤝 [Philémon Vittet](https://github.com/PhilV1tt) — vLLM backend integration

**Things I learned on this one** 🧠:
- 🐳 How Docker layer caching breaks silently when `pip install -U` upgrades torch under your feet (torchao 0.17 → torch 2.12 → CUDA 13 → everything dies)
- 🔋 How **CodeCarbon** tracks energy per-source and why `samples_per_joule` is the real PEReN KPI, not raw accuracy
- 🦙 How `llama-cpp-python` multimodal works internally (`mmproj` projector + image data-URI in chat completions)
- 📦 How to ship an auto-suffisant Docker image via GitHub Actions on GHCR with `ldconfig` for NVIDIA libs

Have fun benchmarking 🔬⚡ !

Code created by [Philémon Vittet](https://github.com/PhilV1tt) and me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)
