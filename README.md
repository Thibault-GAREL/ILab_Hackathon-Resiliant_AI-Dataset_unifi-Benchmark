# Resilient AI Benchmark

Repo d'équipe pour le challenge PEReN Resilient AI à l'ECE ILAB.

Le but est simple : construire un dataset VLM unifié, lancer des benchmarks
comparables sur Gemma 4 / Gemma 3n, puis trier les runs par efficience
énergétique. Après le seuil de qualité imposé par PEReN, le ranking se joue
surtout sur `samples_per_joule`.

## Ce que fait le repo

1. Dataset unifié : 12 sources vision langage dans
   `ECE-ILAB/resilient-ai-unified`.
2. Benchmark par source : DocVQA, VQAv2, ChartQA, COCO Captions, AI2D, MMMU,
   et les autres sources du mix.
3. Métriques adaptées : ANLS, VQA accuracy, CIDEr, multiple choice,
   relaxed accuracy.
4. Tracking énergie : CodeCarbon par source, puis agrégation dans le registry.
5. Backends modèles : `transformers`, `llama_cpp`, `vllm`.

## État actuel

Le dataset HF contient environ 423k samples. Sur Runpod, le dataset local de
référence est attendu dans `/workspace/code/data/unified`.

Le registry principal est `results/registry.csv`. Il est append only : une
ligne par run, avec les métriques qualité, énergie et config.

Les runs par défaut utilisent `country_iso=CZE`, car le pod de référence est
sur Runpod EU CZ 1. Pour une autre région, changer le pays dans la CLI ou le
YAML.

`bnb-4bit` est retiré du YAML. Le bug FP4 sur Gemma 4 multimodal casse
l'initialisation côté `bitsandbytes`. Les variantes AWQ et FP8 couvrent le cas
compression sans garder cette dépendance fragile dans le flow principal.

## Structure

```text
configs/
  models.yaml                 # modèles et groupes de benchmark
src/
  config.py                   # specs dataset
  data/                       # loaders + schema + mix
  models/                     # backends llama_cpp et vllm
  validation/                 # evaluate_model + métriques
  tracking/                   # CodeCarbon + registry
scripts/
  build_dataset.py            # build + push du dataset unifié
  eval_loop.py                # benchmark d'un modèle
  bench_all.py                # batch runner depuis models.yaml
  bench                         # wrapper CLI local
  bench_cli.py                # CLI Typer
  infer_vllm_image.py         # test image direct via label vLLM
results/
  registry.csv                # source de vérité des runs
  runs/                       # artefacts détaillés, gitignored
reports/                      # JSON par run, gitignored
```

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

Sur Mac ou machine sans GPU CUDA, garder les commandes de benchmark en
`--dry-run`. Les vrais runs sont prévus pour Runpod ou une machine CUDA.

## CLI benchmark

La CLI est un front léger au dessus de `scripts/bench_all.py`. Elle lit
`configs/models.yaml`, affiche le plan de run, puis délègue au runner existant.

Setup rapide :

```bash
python3 -m venv /tmp/bench-cli-venv
source /tmp/bench-cli-venv/bin/activate
pip install typer rich questionary PyYAML

PYTHON=/tmp/bench-cli-venv/bin/python scripts/bench doctor
PYTHON=/tmp/bench-cli-venv/bin/python scripts/bench menu
```

Commandes utiles :

```bash
scripts/bench explain
scripts/bench sources
scripts/bench list --backend vllm
scripts/bench model gemma4-e4b-vllm-fp8-kvfp8-soft140-t16
scripts/bench smoke vllm --dry-run
scripts/bench group vllm_submission_candidate --limit 20 --dry-run
scripts/bench try-vllm google/gemma-4-E4B-it --max-soft-tokens 140 --quantization fp8 --kv-cache-dtype fp8 --dry-run
scripts/bench try-gguf bartowski/google_gemma-4-E4B-it-GGUF '*Q4_K_M.gguf' --mmproj-repo bartowski/google_gemma-4-E4B-it-GGUF --mmproj-filename 'mmproj-*f16.gguf' --dry-run
scripts/bench results --top 10
```

## Backends et groupes utiles

| Groupe | Usage |
|---|---|
| `vllm_smoke` | smoke test vLLM minimal |
| `vllm_submission_candidate` | BF16 soft140 + FP8/KV FP8 soft140 |
| `vllm_submission_fallback` | soft280 si soft140 dégrade trop |
| `vllm_runtime_grid` | petite grille runtime vLLM |
| `vllm_full_grid` | grille vLLM complète sur le même artefact HF |
| `vllm_awq_probe` | probes AWQ chargées via vLLM |
| `gguf_bartowski_l4` | grille GGUF Bartowski |
| `gguf_axel_l4` | grille GGUF AxelBrons |
| `gguf_l4_all` | Bartowski + AxelBrons |

Les entrées `vllm` ont `run_by_default: false`. Il faut les lancer avec
`--label` ou `--group`, sinon le batch standard reste sur les modèles par
défaut.

## Runpod

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

Toujours passer `--dataset_local` si le dataset existe déjà sur le volume.
Sinon le runner repart sur le Hub et peut retélécharger plusieurs dizaines de
GB.

Pour lancer le wrapper complet :

```bash
cd /app
nohup bash scripts/bench_safe.sh > /workspace/bench.log 2>&1 &
tail -f /workspace/bench.log
```

Pour lire le registry trié par énergie :

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

Pour récupérer le registry :

```bash
runpodctl send /workspace/results/registry.csv
```

## Serveur vLLM

Deux configs locales permettent de servir les artefacts Scarbote en OpenAI
compatible API :

```bash
bash scripts/serve_vllm_two_models.sh
```

Ports :

| Port | Modèle servi |
|---|---|
| `8000` | `modele-complet-fp8-kvfp8-soft140` |
| `8001` | `modele-complet-lora-energie-soft140` |

Chaque config limite la mémoire GPU à `0.45` pour faire tourner les deux
processus sur le même GPU. Pour servir un seul modèle, remonter
`gpu-memory-utilization` à `0.90`.

Test image direct depuis un label vLLM :

```bash
python scripts/infer_vllm_image.py path/to/image.png \
  --label gemma4-e4b-vllm-fp8-kvfp8-soft140-t16 \
  --prompt "Describe this image."
```

## Résultats connus

Exemple de registry sur RTX 4090, Runpod EU CZ 1, `limit_per_source=20` :

| Modèle | macro_avg | J/sample | samples/J | Note |
|---|---:|---:|---:|---|
| `gemma4-e4b-bf16` | 0.3996 | 117.5 | 0.0085 | baseline qualité |
| `gemma4-e2b-bf16` | 0.3569 | 114.2 | 0.0088 | plus petit |
| `gemma4-e4b-awq-cyankiwi-int4` | 0.4080 | 135.0 | 0.0074 | variance possible sur 240 samples |
| `gemma4-e4b-awq-cyankiwi-int8` | 0.3958 | 125.0 | 0.0080 | AWQ moins agressif |

Ces chiffres sont indicatifs. Pour décider, relancer sur le protocole de
soumission et trier par `samples_per_joule` après vérification du seuil qualité.

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
| `--subsample_pct` | `10.0` dans `eval_loop.py` | pourcentage par source |
| `--limit_per_source` | vide | cap pour smoke tests |
| `--country_iso` | `FRA` dans `eval_loop.py`, `CZE` dans le YAML | mix électrique CodeCarbon |
| `--run_label` | dérivé | nom dans le registry |

## Sources

Le repo s'appuie surtout sur :

1. Hugging Face `datasets` pour le chargement unifié.
2. Hugging Face `transformers` pour Gemma multimodal.
3. `llama-cpp-python` pour GGUF + `mmproj`.
4. `vllm` pour les runs FP8 / KV FP8.
5. CodeCarbon pour l'énergie.
6. Quantifications GGUF Bartowski et AxelBrons.
7. Checkpoints AWQ cyankiwi.

Contributeurs visibles dans ce repo : Thibault Garel et Philémon Vittet,
notamment pour l'intégration vLLM.
