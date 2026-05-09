# Resilient AI — Dataset unifié + Benchmark

Code partagé de l'équipe. Trois responsabilités :

1. **Dataset unifié** (12 sources → 1 repo HuggingFace) — `build_dataset.py`
2. **Benchmark interne** (10% du dataset, par-source, métrique adaptée) — `eval_loop.py`
3. **Tracking énergie + traçabilité** (CodeCarbon + registry CSV) — `energy_tracker.py` + `run_registry.py`

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
  reste dans `eval_ood.py` pour usage manuel ponctuel.

## Installation (Runpod ou local)

```bash
pip install -r requirements.txt
huggingface-cli login   # token écriture pour le push du dataset
```

Dépendances clés ajoutées : `codecarbon>=2.4.0`, `nvidia-ml-py>=12.535`,
`psutil>=5.9`. Si `codecarbon` est absent, le tracking dégrade
silencieusement en wall-time-only — utile pour debug local mais à NE
JAMAIS publier.

## Build du dataset sur Runpod (one-shot)

Le build complet charge VQAv2 (~20 GB), COCO (~20 GB), DocVQA, etc. — **à ne
pas lancer sur un laptop**. Pod Runpod ≥32 GB RAM, ≥200 GB disque,
CPU-only OK (moins cher).

```bash
cd code/
export HF_TOKEN=hf_xxxxx        # scope repo.write
bash runpod_run.sh
```

Le script enchaîne : `pip install`, login HF, dry-run, build complet, push.
Variables d'override (cf. entête `runpod_run.sh`) : `HUB_ID`, `TIER`,
`STRATEGY`, `TEMPERATURE`, `TRAIN_SIZE`, `MAX_PER_SOURCE`, `MAX_IMAGE_DIM`,
`OUT_DIR`, `PRIVATE`, `SKIP_INSTALL`.

Hub courant : `ECE-ILAB/resilient-ai-unified` (v0.3, 423k lignes,
12 sources sqrt-balanced + calibration PTQ 2048 samples).

## Exécution du benchmark — entrypoint `eval_loop.py` (recommandé)

```bash
# 1. Smoke test (oracle, ~1 min, doit score ~1.0)
python eval_loop.py \
    --mode oracle \
    --dataset_hub In/resilient-ai-unified \
    --subsample_pct 5 \
    --run_label smoke-oracle \
    --out reports/smoke.json

# 2. Baseline Gemma 4 non compressé
python eval_loop.py \
    --mode real \
    --model_path google/gemma-3n-E4B-it \
    --dataset_hub ECE-ILAB/resilient-ai-unified \
    --run_label gemma3n-e4b-baseline \
    --out reports/baseline.json

# 3. Votre checkpoint compressé (ex: Djebril après QAT INT4)
python eval_loop.py \
    --mode real \
    --model_path runs/djebril-qat-int4/final \
    --dataset_hub ECE-ILAB/resilient-ai-unified \
    --run_label djebril-qat-int4 \
    --notes "QAT 4 epochs, calib 2048 samples" \
    --out reports/djebril-qat-int4.json

# 4. Run final avant soumission — full split
python eval_loop.py \
    --mode real \
    --model_path runs/student-final/final \
    --dataset_hub ECE-ILAB/resilient-ai-unified \
    --subsample_pct 100 \
    --run_label final-full \
    --out reports/final-full.json
```

### Outputs générés à chaque run

```
<out>                                          # rapport JSON principal (résumé + détails)
results/registry.csv                           # APPEND-ONLY, +1 ligne par run (cf. infra)
results/runs/<label>_<ts>.json                 # artefact détaillé full
results/runs/<label>_<ts>_energy.csv           # énergie par-source (1 ligne / source)
results/runs/<label>_<ts>_summary.txt          # résumé human-readable 1 page
```

### Le registry global `results/registry.csv`

C'est notre source de vérité pour comparer toutes les runs. **32 colonnes**
incluant : `timestamp_iso, run_label, git_commit, model_id, subsample_pct,
macro_avg, micro_avg, kwh_total, kwh_gpu, kwh_cpu, kwh_ram, co2_g,
joules_per_sample, samples_per_joule, score_per_kwh, gpu_name, hostname,
notes, …`

Ouvrir directement dans Excel / pandas pour trier par `samples_per_joule`
desc et identifier les checkpoints les plus efficaces. **Append-only** :
ne pas le rebuild, on ajoute simplement les nouvelles runs au bout.

### Args clés `eval_loop.py`

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

## Architecture des modules

```
eval_loop.py        ← entrypoint principal (CLI)
   │
   ├─ benchmark.py        evaluate_model() + subsample_per_source() + adaptateurs
   │     └─ metrics.py    score_sample() (VQA acc / ANLS / MC / CIDEr / Relaxed)
   │
   ├─ energy_tracker.py   EnergyTracker (CodeCarbon) + EnergySnapshot + KPIs
   │
   └─ run_registry.py     RegistryEntry + append_to_registry + dump_run_artifacts

build_dataset.py    ← entrypoint dataset (CLI)
   ├─ config.py           DatasetSpec, ALL_DATASETS, MixConfig
   ├─ loaders.py          load_source() — 12 loaders + UnifiedSample mapping
   ├─ mix.py              build_mix() avec stratégies sqrt / uniform / temperature
   ├─ schema.py           UnifiedSample dataclass + hf_features
   └─ validate_dataset.py post-build sanity check

eval_ood.py         ← USAGE MANUEL UNIQUEMENT (MMStar/MME/POPE)
                      n'est plus appelé par eval_loop.py
```

## Métriques par source (inchangé)

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
