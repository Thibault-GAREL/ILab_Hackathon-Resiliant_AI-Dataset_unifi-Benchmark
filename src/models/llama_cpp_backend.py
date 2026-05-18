"""
Backend llama-cpp-python pour les modèles GGUF (bartowski, unsloth UD-*, ggml-org).

⚠️ EXPÉRIMENTAL — le support multimodal de Gemma 3n / Gemma 4 dans llama.cpp
est récent (mid-2025) et peut nécessiter une version récente de
`llama-cpp-python` + le bon `mmproj-*.gguf` co-localisé avec le modèle.

Le pipeline transformers reste la référence pour les comparaisons officielles
de l'équipe ; ce backend permet d'évaluer la grille de quantifications GGUF
publiée par bartowski/unsloth sans avoir à les reconvertir en safetensors.

Usage (depuis benchmark.py via eval_loop.py --backend llama_cpp) :

    gen_fn = make_llama_cpp_generate_fn(
        gguf_repo="bartowski/gemma-3n-E4B-it-GGUF",
        gguf_filename="*Q4_K_M.gguf",
        mmproj_repo="bartowski/gemma-3n-E4B-it-GGUF",
        mmproj_filename="mmproj-*.gguf",
        max_new_tokens=32,
        n_gpu_layers=-1,
        n_ctx=2048,
    )

Tous les arguments `*_repo` / `*_filename` peuvent aussi être des chemins
locaux : si le path existe sur disque, on l'utilise tel quel sans passer
par le hub HF.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Callable

GenerateFn = Callable[..., str]


def _image_to_data_uri(image, format: str = "PNG") -> str:
    """Encode une PIL Image en data-URI base64 — format attendu par llama.cpp
    pour l'API multimodale OpenAI-compatible."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{encoded}"


def _resolve_chat_handler(model_family: str | None, mmproj_path: str):
    """Sélectionne le ChatHandler llama.cpp adapté à la famille de modèle.

    Pour Gemma 3 / 3n / 4, on utilise `Gemma3ChatHandler` (introduit dans
    llama-cpp-python ≥ 0.3.x). Si le handler n'existe pas dans la version
    installée, on fall-back sur `Llava15ChatHandler` (générique) — ça peut
    fonctionner pour des familles non Gemma mais pas garanti pour Gemma 3n.
    """
    from llama_cpp.llama_chat_format import (
        Llava15ChatHandler,
    )

    family = (model_family or "gemma3").lower()

    # Tentative d'import du handler natif Gemma3 (peut ne pas exister selon
    # la version de llama-cpp-python).
    if family.startswith("gemma"):
        try:
            from llama_cpp.llama_chat_format import Gemma3ChatHandler
            return Gemma3ChatHandler(clip_model_path=mmproj_path, verbose=False)
        except ImportError:
            pass

    # Fall-back générique LLaVA 1.5 (peut suffire pour des modèles compatibles
    # avec ce template, sinon adapter ici).
    return Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)


def _resolve_path(repo: str, filename: str) -> str:
    """Si `repo` est un dossier local existant, joint avec `filename` ;
    sinon télécharge le fichier depuis le hub HF et retourne le chemin local.

    Supporte les globs (`*`, `?`, `[...]`) dans `filename` : on liste les
    fichiers du repo, on garde le premier match. Indispensable car
    `hf_hub_download` ne gère pas les globs et tape une URL littérale
    (ex: `mmproj-*f16.gguf` → 404).
    """
    import fnmatch
    repo_path = Path(repo)

    is_glob = any(c in filename for c in ["*", "?", "["])

    # Cas local : dossier sur disque
    if repo_path.is_dir():
        if is_glob:
            matches = sorted(repo_path.glob(filename))
            if not matches:
                raise FileNotFoundError(
                    f"Aucun fichier ne match {filename!r} dans {repo}"
                )
            return str(matches[0])
        local = repo_path / filename
        if not local.exists():
            raise FileNotFoundError(f"{local} introuvable dans {repo}")
        return str(local)

    # Cas hub HF
    from huggingface_hub import hf_hub_download, HfApi

    if is_glob:
        api = HfApi()
        try:
            files = api.list_repo_files(repo_id=repo)
        except Exception as e:
            raise RuntimeError(
                f"Impossible de lister les fichiers de {repo}: {e}"
            ) from e
        matches = [f for f in files if fnmatch.fnmatch(f, filename)]
        if not matches:
            # Suggère les ggufs disponibles pour aider le debug
            ggufs = [f for f in files if f.endswith(".gguf")][:10]
            raise FileNotFoundError(
                f"Aucun fichier ne match {filename!r} dans {repo}.\n"
                f"GGUF disponibles (10 premiers) : {ggufs}"
            )
        resolved = sorted(matches)[0]
        print(f"   [resolve] glob {filename!r} → {resolved}")
        filename = resolved

    return hf_hub_download(repo_id=repo, filename=filename)


def make_llama_cpp_generate_fn(
    gguf_repo: str,
    gguf_filename: str,
    mmproj_repo: str | None = None,
    mmproj_filename: str | None = None,
    max_new_tokens: int = 32,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    model_family: str | None = "gemma3",
    seed: int = 42,
) -> GenerateFn:
    """Charge un modèle GGUF via llama-cpp-python et retourne une GenerateFn
    compatible avec evaluate_model() (même signature que make_gemma_generate_fn).

    Arguments
    ---------
    gguf_repo, gguf_filename : repo HF + nom de fichier (ou chemin local du
        dossier + nom du .gguf) du modèle principal.
    mmproj_repo, mmproj_filename : idem pour le multimodal projector. Requis
        pour les modèles VLM (Gemma 3n / 4 multimodaux). Si None, le modèle
        est chargé en text-only (échouera sur les samples avec image).
    max_new_tokens : limite de tokens générés (équivalent à transformers).
    n_gpu_layers : -1 = toutes les couches sur GPU (recommandé sur RTX/A100).
        0 = CPU only. Override pour fit dans une VRAM limitée.
    n_ctx : taille du contexte (4096 pour les Gemma 3n, 2048 suffit pour le
        bench où les prompts sont courts).
    model_family : "gemma3" / "gemma4" / "llava" — pilote le ChatHandler.
    """
    from llama_cpp import Llama

    print(f"[load gguf] {gguf_repo}/{gguf_filename}")
    if mmproj_repo and mmproj_filename:
        print(f"[load mmproj] {mmproj_repo}/{mmproj_filename}")
        mmproj_path = _resolve_path(mmproj_repo, mmproj_filename)
        chat_handler = _resolve_chat_handler(model_family, mmproj_path)
    else:
        chat_handler = None

    # Llama.from_pretrained gère HF Hub + cache automatiquement si on passe
    # un repo_id. Pour un chemin local, on bascule sur Llama() direct.
    if Path(gguf_repo).is_dir():
        model_path = _resolve_path(gguf_repo, gguf_filename)
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            seed=seed,
            verbose=False,
        )
    else:
        llm = Llama.from_pretrained(
            repo_id=gguf_repo,
            filename=gguf_filename,
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            seed=seed,
            verbose=False,
        )

    def _gen(image, prompt, sample=None, **kwargs) -> str:
        if image is not None:
            content = [
                {"type": "image_url",
                 "image_url": {"url": _image_to_data_uri(image)}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=max_new_tokens,
            temperature=0.0,  # équivalent do_sample=False dans transformers
        )
        return response["choices"][0]["message"]["content"].strip()

    return _gen
