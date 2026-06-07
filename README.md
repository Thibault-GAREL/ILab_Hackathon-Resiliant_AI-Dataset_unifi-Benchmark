# Resilient AI Benchmark

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
  <img src="https://github.com/Thibault-GAREL/Thibault-GAREL/raw/main/Logo_Group_Projects/group_resilient_sq.png" alt="Logo Resilient AI Challenge" width="200">
</p>

## Description

Ce repo contient le benchmark d'équipe pour le challenge PEReN Resilient AI à l'ECE ILAB.

Le pipeline construit un dataset vision langage unifié, puis évalue des modèles Gemma 4 et Gemma 3n avec des métriques par source et un suivi énergie CodeCarbon. Le but est de comparer les modèles sur le même protocole, surtout en `J/sample` et `samples/J`, parce que le ranking PEReN est dominé par l'énergie une fois le seuil qualité atteint.

## Fonctionnalités

| Bloc | Rôle |
|---|---|
| Dataset unifié | 12 sources vision langage dans `ECE-ILAB/resilient-ai-unified` |
| Métriques par source | ANLS, VQA accuracy, CIDEr, multiple choice, relaxed accuracy |
| Énergie | CodeCarbon par source, puis agrégation en fin de run |
| Registry | `results/registry.csv`, append-only, une ligne par run |
| Backends | `transformers`, `llama_cpp`, `vllm` |
| RunPod | image Docker GHCR, cache HF sur `/workspace`, cleanup optionnel |

## Structure

```text
configs/
  models.yaml                 # modèles et groupes de benchmark
src/
  config.py                   # specs dataset
  data/                       # loaders, schema, mix
  models/                     # backends llama_cpp et vllm
  validation/                 # benchmark + métriques
  tracking/                   # CodeCarbon + registry
scripts/
  build_dataset.py            # build + push du dataset unifié
  eval_loop.py                # benchmark d'un modèle
  bench_all.py                # batch runner depuis models.yaml
  bench                         # wrapper CLI local
  bench_cli.py                # CLI Typer
  bench_safe.sh               # wrapper RunPod
results/
  registry.csv                # source de vérité des runs
  runs/                       # artefacts détaillés
reports/                      # rapports JSON par run
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

Sur une machine sans GPU CUDA, garder les commandes de benchmark en `--dry-run`. Les vrais runs sont prévus pour RunPod ou une machine CUDA.

## Commandes utiles

Lister les modèles et groupes disponibles :

```bash
python scripts/bench_all.py --list
```

Lancer un modèle :

```bash
python scripts/bench_all.py \
  --label gemma4-e4b-vllm-fp8-kvfp8-soft140-t16 \
  --dataset_local /workspace/code/data/unified \
  --limit_per_source 20 \
  --country_iso CZE
```

Lancer un groupe :

```bash
python scripts/bench_all.py \
  --group vllm_submission_candidate \
  --dataset_local /workspace/code/data/unified \
  --limit_per_source 20 \
  --cleanup_cache_after \
  --skip_existing \
  --country_iso CZE
```

Dry-run local :

```bash
python scripts/bench_all.py \
  --group vllm_smoke \
  --dry_run \
  --dataset_hub ECE-ILAB/resilient-ai-unified \
  --dataset_streaming_per_source 2 \
  --limit_per_source 2 \
  --country_iso CZE
```

## Groupes principaux

| Groupe | Usage |
|---|---|
| `jury_like_anchor` | modèle officiel soumis `Mathos34400/resilient-challenge-image-to-text` |
| `jury_like_compare` | ancre Mathos, baseline BF16, candidat vLLM actuel |
| `vllm_smoke` | smoke test vLLM minimal |
| `vllm_submission_candidate` | BF16 soft140 + FP8/KV FP8 soft140 |
| `vllm_submission_fallback` | soft280 si soft140 dégrade trop |
| `vllm_runtime_grid` | petite grille runtime vLLM |
| `vllm_full_grid` | grille vLLM complète |
| `vllm_awq_probe` | probes AWQ chargées via vLLM |
| `gguf_bartowski_l4` | grille GGUF Bartowski |
| `gguf_axel_l4` | grille GGUF AxelBrons |
| `gguf_l4_all` | Bartowski + AxelBrons |

Les entrées `vllm` ont `run_by_default: false`. Il faut les lancer avec `--label` ou `--group`.

## Sortie format jury

Chaque run écrit deux artefacts au format du tableau jury :

```text
results/runs/<label>_<timestamp>_leaderboard_like.csv
results/runs/<label>_<timestamp>_leaderboard_like.md
```

Le CSV garde les colonnes exactes du jury :

```text
team_name,energy_consumption,document_analysis,image_understanding,mean,mean_recovery
```

Les trois premières lignes sont les ancres officielles Round 1 :

| team_name | energy_consumption | document_analysis | image_understanding | mean | mean_recovery |
|---|---:|---:|---:|---:|---:|
| base_model | 99.70575476453993 | 0.7607534745955797 | 0.68 | 0.7203767372977898 | 100.0 |
| Top model from Round 1 | 35.048636372219946 | 0.5634310134310134 | 0.7 | 0.6317155067155067 | 87.69238011281973 |
| Intelligence Lab ECE | 205.91259250771995 | 0.8103779553779553 | 0.5869565217391305 | 0.6986672385585428 | 96.9863687130318 |

Calcul local :

```text
document_analysis = mean(docvqa, textvqa, chartqa, infographicvqa)
image_understanding = mean(vqav2, ok_vqa, coco_captions, realworldqa, scienceqa, ai2d, mathvista, mmmu)
mean = (document_analysis + image_understanding) / 2
mean_recovery = mean / 0.7203767372977898 * 100
energy_consumption = CodeCarbon J/sample
```

Smoke de l'ancre Mathos :

```bash
python scripts/bench_all.py \
  --group jury_like_anchor \
  --dataset_hub ECE-ILAB/resilient-ai-unified \
  --dataset_streaming_per_source 1 \
  --limit_per_source 1 \
  --country_iso CZE
```

## RunPod

Image Docker publique :

```bash
ghcr.io/thibault-garel/ilab_hackathon-resiliant_ai-dataset_unifi-benchmark:latest
```

Reprise après démarrage d'un pod :

```bash
export HF_HOME=/workspace/hf-cache
cd /app

python scripts/bench_all.py \
  --group vllm_submission_candidate \
  --dataset_local /workspace/code/data/unified \
  --limit_per_source 20 \
  --cleanup_cache_after \
  --skip_existing \
  --country_iso CZE
```

Toujours passer `--dataset_local` si le dataset existe déjà sur le volume. Sinon le runner repart sur le Hub et peut retélécharger plusieurs dizaines de GB.

Pour lancer le wrapper complet :

```bash
cd /app
nohup bash scripts/bench_safe.sh > /workspace/bench.log 2>&1 &
tail -f /workspace/bench.log
```

Lire le registry trié par efficacité :

```bash
python -c "
import csv
with open('/workspace/results/registry.csv') as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r.get('samples_per_joule', 0) or 0), reverse=True)
print(f'{len(rows)} runs')
print(f'{\"label\":<42} {\"macro\":>8} {\"J/sample\":>10} {\"samples/J\":>12}')
for r in rows:
    print(f'{r[\"run_label\"]:<42} {r[\"macro_avg\"]:>8} {r[\"joules_per_sample\"]:>10} {r[\"samples_per_joule\"]:>12}')
"
```

Récupérer le registry :

```bash
runpodctl send /workspace/results/registry.csv
```

## Flags importants

| Flag | Défaut | Usage |
|---|---|---|
| `--backend` | `transformers` | `transformers`, `llama_cpp`, `vllm` |
| `--model_path` | vide | requis pour `transformers` et `vllm` |
| `--gguf_repo` | vide | repo HF ou chemin local GGUF |
| `--gguf_filename` | vide | glob GGUF côté Hub |
| `--mmproj_repo` | vide | projector multimodal GGUF |
| `--dataset_local` | vide | dataset `save_to_disk` local |
| `--dataset_hub` | vide | dataset HF |
| `--dataset_streaming_per_source` | vide | lit N samples par source en streaming |
| `--subsample_pct` | `10.0` dans `eval_loop.py` | pourcentage par source |
| `--limit_per_source` | vide | cap pour smoke tests |
| `--country_iso` | `FRA` dans `eval_loop.py`, `CZE` dans le YAML | mix électrique CodeCarbon |
| `--run_label` | dérivé | nom dans le registry |
| `--out` | requis | rapport JSON principal |

## Résultats connus

Exemple de registry sur RTX 4090, RunPod EU CZ 1, `limit_per_source=20` :

| Modèle | macro_avg | J/sample | samples/J | Note |
|---|---:|---:|---:|---|
| `gemma4-e4b-bf16` | 0.3996 | 117.5 | 0.0085 | baseline qualité |
| `gemma4-e2b-bf16` | 0.3569 | 114.2 | 0.0088 | modèle plus petit |
| `gemma4-e4b-awq-cyankiwi-int4` | 0.4080 | 135.0 | 0.0074 | variance possible sur 240 samples |
| `gemma4-e4b-awq-cyankiwi-int8` | 0.3958 | 125.0 | 0.0080 | AWQ moins agressif |

Ces chiffres sont indicatifs. Pour décider, relancer sur le protocole de soumission et comparer la sortie format jury.

## Sources

Le repo s'appuie surtout sur :

1. Hugging Face `datasets` pour le chargement unifié.
2. Hugging Face `transformers` pour Gemma multimodal.
3. `llama-cpp-python` pour GGUF + `mmproj`.
4. `vllm` pour les runs FP8 / KV FP8.
5. CodeCarbon pour l'énergie.
6. Quantifications GGUF Bartowski et AxelBrons.
7. Checkpoints AWQ cyankiwi.

## Contributeurs

Code créé par Thibault Garel et Philémon Vittet pour le challenge Resilient AI.

Philémon a intégré le backend vLLM, les variantes FP8 / KV FP8, les budgets visuels `soft140` / `soft280`, le flow RunPod associé, et la sortie format jury pour comparer les runs locaux aux ancres officielles.
