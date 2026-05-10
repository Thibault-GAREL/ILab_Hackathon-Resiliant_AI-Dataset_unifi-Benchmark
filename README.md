# Resilient AI — Dataset unifié + Benchmark

Code partagé de l'équipe. Trois responsabilités :

1. **Dataset unifié** (12 sources → 1 repo HuggingFace) — `scripts/build_dataset.py`
2. **Benchmark interne** (10% du dataset, par-source, métrique adaptée) — `scripts/eval_loop.py`
3. **Tracking énergie + traçabilité** (CodeCarbon + registry CSV) — `src/tracking/`

> Le code cible Gemma 3n E4B → Student E4B-it (config officielle kickoff
> 27/04). Pour Gemma 4 quand l'API se stabilisera, bumper `transformers` et
> ajuster `make_gemma_generate_fn` si l'API du processor change.

## Pourquoi c'est structuré comme ça

Le ranking PEReN est dominé par l'énergie (CodeCarbon) **après** le seuil
80% sur les 7 benchmarks publics. Conséquence :

- Toutes les runs benchmark sont trackées CodeCarbon par défaut, par-source,
  et appendées dans un registry CSV global pour pouvoir comparer 50 runs
  sans perdre l'historique.
- Le bench tourne par défaut sur **10% du dataset unifié** (subsample
  déterministe seed=42) — itération rapide. Override `--subsample_pct 100`
  pour le full split avant soumission finale.
- Pas d'évaluation OOD (MMStar/MME/POPE) dans le flow par défaut. Le code
  reste dans `src/validation/eval_ood.py` pour usage manuel ponctuel.

## Architecture du projet

```
.
├── configs/
│   └── models.yaml             # Liste des modèles à benchmarker
│
├── src/                        # Code métier (importable depuis scripts/)
│   ├── config.py               # DatasetSpec, ALL_DATASETS, MixConfig
│   ├── data/
│   │   ├── schema.py           # UnifiedSample dataclass + hf_features
│   │   ├── loaders.py          # 12 loaders + UnifiedSample mapping
│   │   └── mix.py              # build_mix() avec stratégies sqrt/uniform/temperature
│   ├── validation/
│   │   ├── benchmark.py        # evaluate_model() + subsample_per_source()
│   │   ├── metrics.py          # score_sample (VQA/ANLS/MC/CIDEr/Relaxed)
│   │   └── eval_ood.py         # Usage manuel uniquement (MMStar/MME/POPE)
│   └── tracking/
│       ├── energy_tracker.py   # EnergyTracker (CodeCarbon) + KPIs
│       └── run_registry.py     # RegistryEntry + append_to_registry
│
├── scripts/                    # Entrypoints CLI
│   ├── build_dataset.py        # Build + push du dataset unifié
│   ├── eval_loop.py            # Benchmark d'un modèle (entrypoint principal)
│   ├── bench_all.py            # Orchestrateur batch (lit configs/models.yaml)
│   ├── run_benchmark.py        # Variante historique (préfère eval_loop.py)
│   ├── validate_dataset.py     # Sanity check post-build
│   ├── runpod_setup.sh         # Setup d'un pod Runpod (one-shot)
│   └── runpod_run.sh           # Build dataset sur Runpod
│
├── data/                       # Données locales (gitignored)
│   ├── 1-raw/                  # Données brutes — NEVER modify
│   ├── 2-processed/            # Données nettoyées
│   └── 3-external/             # APIs / datasets publics
│
├── reports/                    # Rapports JSON par run (gitignored)
├── results/                    # Registry global + artefacts détaillés
│   ├── registry.csv            # APPEND-ONLY, 1 ligne / run
│   └── runs/                   # JSON + CSV détaillés par run
│
├── tests/
├── requirements.txt
└── README.md
```

## Installation

### Local (Windows + venv `pytorch_cuda_env`)

```powershell
& c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
pip install -r requirements.txt
huggingface-cli login
```

### Sur Runpod — option A : Docker (recommandée, reproductible)

L'image embarque PyTorch + CUDA 12.1 + transformers + bitsandbytes + autoawq +
llama-cpp-python (CUDA wheel pré-compilé). Volumes pour persister cache HF et
résultats.

```powershell
# 1) Build local + push sur Docker Hub (à faire une fois) :
docker build -t <ton-user>/resilient-ai:latest .
docker push <ton-user>/resilient-ai:latest

# 2) Sur Runpod :
#    Deploy a Pod → Custom Template → Container image: <ton-user>/resilient-ai:latest
#    Volume mount: /cache/huggingface (10-30 GB pour cache modèles)
#    Volume mount: /app/results (~1 GB pour registry + artefacts)
#    Env: HF_TOKEN=hf_xxx
#
#    Au démarrage, le container exécute "python scripts/bench_all.py --list".
#    Override la commande avec :
#      python scripts/bench_all.py --label gemma3n-e4b-bf16
#      python scripts/bench_all.py --all --skip_existing
```

Pour itérer en local sur la GTX 1660 Ti (6 GB VRAM, donc forcément modèle
quantifié) :

```powershell
$env:HF_TOKEN = "hf_xxx"
docker compose build
docker compose run --rm bench python scripts/bench_all.py --label gemma4-e2b-bnb4
```

### Sur Runpod — option B : sans Docker (pip install à chaque pod)

```bash
git clone <repo-url> && cd ILab_Resilient_AI_*
bash scripts/runpod_setup.sh
```

Le script installe `requirements.txt`, `bitsandbytes`, `autoawq`, login HF, et lance un smoke test oracle.

## Workflow benchmark sur Runpod

### Étape 1 — Lancer un pod
- **GPU** : A100 40 GB (~1 €/h) ou RTX 4090 24 GB
- **Disk** : 100 GB minimum (cache HF + dataset stream)
- **Image** : `<ton-user>/resilient-ai:latest` (option A) ou
  `runpod/pytorch:2.5.1-py3.11-cuda12.1.1-devel-ubuntu22.04` (option B)

### Étape 2 — Setup du pod (option B uniquement)

```bash
bash scripts/runpod_setup.sh
```

### Étape 3 — Lister les modèles disponibles

```bash
python scripts/bench_all.py --list
```

Sortie :
```
LABEL                        EXTRAS         MODEL_PATH
------------------------------------------------------------------------------------------
gemma4-e4b-bf16              -              google/gemma-4-E4B-it
gemma3n-e4b-bf16             -              google/gemma-3n-E4B-it
gemma4-e4b-bnb4              bitsandbytes   unsloth/gemma-4-E4B-it-bnb-4bit
gemma4-e4b-bnb4-dyn          bitsandbytes   unsloth/gemma-4-E4B-it-unsloth-bnb-4bit
gemma4-e4b-awq4              autoawq        cyankiwi/gemma-4-E4B-it-AWQ-4bit
gemma4-e2b-bf16              -              google/gemma-4-E2B-it
gemma4-e2b-bnb4              bitsandbytes   unsloth/gemma-4-E2B-it-bnb-4bit
gemma4-e2b-bnb4-dyn          bitsandbytes   unsloth/gemma-4-E2B-it-unsloth-bnb-4bit
```

### Étape 4 — Benchmarker

```bash
# Un seul modèle (recommandé pour itérer)
python scripts/bench_all.py --label gemma4-e4b-bf16

# Sous-ensemble
python scripts/bench_all.py --label gemma4-e4b-bf16 --label gemma4-e4b-bnb4

# Tous d'affilée (laisse tourner la nuit)
python scripts/bench_all.py --all

# Run final avant soumission — full split au lieu de 10%
python scripts/bench_all.py --all --subsample_pct 100

# Reprendre après un échec sans tout refaire
python scripts/bench_all.py --all --skip_existing
```

### Étape 5 — Récupérer les résultats

```bash
# Depuis ta machine locale :
scp -r runpod:/workspace/.../results ./results-from-pod/
scp -r runpod:/workspace/.../reports ./reports-from-pod/
```

Puis ouvre `results/registry.csv` dans Excel ou pandas pour trier par
`samples_per_joule` desc.

## Modifier la liste de modèles

Édite [configs/models.yaml](configs/models.yaml). Format selon le backend :

### Backend `transformers` (défaut)

```yaml
models:
  - label: mon-checkpoint-v1     # nom court → registry + reports/<label>.json
    model_path: user/repo-id     # id HuggingFace OU chemin local
    notes: "Description courte"
    extras: bitsandbytes         # optionnel : dépendance pip à installer
```

### Backend `llama_cpp` (GGUF multimodal)

```yaml
models:
  - label: mon-gguf-q4km
    backend: llama_cpp
    gguf_repo: bartowski/mon-modele-GGUF       # repo HF (ou chemin local)
    gguf_filename: "*Q4_K_M.gguf"               # glob accepté côté HF Hub
    mmproj_repo: bartowski/mon-modele-GGUF      # multimodal projector
    mmproj_filename: "mmproj-*f16.gguf"
    model_family: gemma4                        # gemma3 / gemma4 / llava
    notes: "GGUF Q4_K_M via llama.cpp"
```

### Compatibilité formats

| Format | Backend | Statut |
|---|---|---|
| Safetensors BF16 (Google officiel) | `transformers` | ✅ Stable |
| bitsandbytes 4-bit (`*-bnb-4bit`) | `transformers` + `bitsandbytes` | ✅ Stable |
| AWQ (`*-AWQ-4bit`) | `transformers` + `autoawq` | ✅ Stable |
| GGUF imatrix (bartowski Q*/IQ*) | `llama_cpp` | ⚠️ Expérimental |
| GGUF Dynamic (unsloth UD-*) | `llama_cpp` | ⚠️ Expérimental |
| MLX | – | ❌ Apple Silicon only |
| UQFF | – | ❌ Runtime `mistral.rs` séparé |

⚠️ **Le multimodal Gemma en GGUF dépend de la version de `llama-cpp-python`**.
Le Dockerfile installe le wheel CUDA 12.1 le plus récent, mais le support
`Gemma3ChatHandler` peut nécessiter une montée de version. Si un modèle GGUF
plante au chargement, le backend `transformers` reste la référence pour
l'équipe.

## Build du dataset (one-shot, ~20 GB+)

```bash
cd ILab_Resilient_AI_*
export HF_TOKEN=hf_xxxxx        # scope repo.write
bash scripts/runpod_run.sh
```

Variables d'override (cf. entête `runpod_run.sh`) : `HUB_ID`, `TIER`,
`STRATEGY`, `TEMPERATURE`, `TRAIN_SIZE`, `MAX_PER_SOURCE`, `MAX_IMAGE_DIM`,
`OUT_DIR`, `PRIVATE`, `SKIP_INSTALL`.

Hub courant : `ECE-ILAB/resilient-ai-unified` (v0.3, 423k lignes,
12 sources sqrt-balanced + calibration PTQ 2048 samples).

## Outputs générés à chaque run

```
reports/<label>.json                           # rapport JSON principal
results/registry.csv                           # APPEND-ONLY, +1 ligne / run
results/runs/<label>_<ts>.json                 # artefact détaillé full
results/runs/<label>_<ts>_energy.csv           # énergie par-source
results/runs/<label>_<ts>_summary.txt          # résumé human-readable
```

### Le registry global `results/registry.csv`

Source de vérité pour comparer toutes les runs. **32 colonnes** incluant :
`timestamp_iso, run_label, git_commit, model_id, subsample_pct, macro_avg,
micro_avg, kwh_total, kwh_gpu, kwh_cpu, kwh_ram, co2_g, joules_per_sample,
samples_per_joule, score_per_kwh, gpu_name, hostname, notes, …`

Append-only : ne pas le rebuild, on ajoute simplement les nouvelles runs au bout.

## Args clés `eval_loop.py`

| Flag | Défaut | Utilité |
|---|---|---|
| `--mode` | `real` | `real` / `oracle` / `constant` (debug) |
| `--model_path` | – | requis en mode real |
| `--dataset_hub` ou `--dataset_local` | – | l'un des deux requis |
| `--split` | `test` | `validation` / `test` |
| `--subsample_pct` | `10.0` | % par-source à évaluer (100 = full) |
| `--subsample_seed` | `42` | seed du subsample (reproductibilité inter-runs) |
| `--sources` | tout | liste blanche pour focus debug |
| `--limit_per_source` | – | cap dur post-subsample (smoke test) |
| `--country_iso` | `FRA` | code pays CodeCarbon (override Runpod US) |
| `--no_energy_tracking` | off | DEBUG uniquement, ne JAMAIS publier sans énergie |
| `--run_label` | dérivé | identifiant court dans le registry |
| `--notes` | `""` | texte libre stocké dans le registry |
| `--registry_csv` | `results/registry.csv` | chemin du registry |
| `--out` | requis | rapport JSON principal |

## Métriques par source

| Source | Métrique | Pourquoi |
|---|---|---|
| DocVQA | ANLS (seuil 0.5) | Tolérance OCR aux petites erreurs |
| VQAv2, TextVQA, OK-VQA | VQA accuracy (consensus 10) | Standard officiel |
| MMMU, AI2D, ScienceQA | Multiple choice (lettre) | QCM |
| ChartQA | Relaxed accuracy (±5% num) | Valeurs numériques tolérantes |
| RealWorldQA | Realworld accuracy | Mix MC / ouvert |
| COCO Captions | CIDEr | Standard captioning |
| ST-VQA, OCR-VQA | ANLS | OCR-heavy |

Agrégats : `micro_avg` (pondéré par taille) + `macro_avg` (poids égal par
source). **macro_avg = indicateur principal challenge** car un modèle
compressé peut s'effondrer sur DocVQA (OCR) sans être pénalisé en micro si
VQAv2 reste bon.

## Reproductibilité

- Seed = **42** partout (build dataset, subsample bench, init torch).
- Subsample 10% = déterministe via `seed XOR hash(source)` → deux runs sur
  le même split tirent les **mêmes** samples → comparaisons valides.
- `git_commit` (12 premiers chars) écrit dans chaque ligne du registry
  pour tracer le code utilisé.
- Hardware fingerprint (GPU, hostname, RAM) écrit dans chaque snapshot
  pour expliquer les variations de kWh entre runs sur des pods différents.

## Rappels obligatoires pour publier un résultat équipe

1. **Toujours** activer le tracking énergie. Pas de `--no_energy_tracking`.
2. **Toujours** mettre un `--run_label` parlant (`djebril-qat-int4-v3`,
   `axel-pruning-50pct`) — c'est ce qu'on verra dans le registry.
3. Avant la soumission finale : run sur `--subsample_pct 100` pour valider
   sur le full split, **pas** sur 10%.
4. Pour comparer deux runs : ne pas se fier au JSON, ouvrir
   `results/registry.csv` et comparer côte-à-côte. Les KPIs énergie sont
   plus stables que les scores absolus.
