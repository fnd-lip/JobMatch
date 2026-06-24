from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any

import cloudpickle
from huggingface_hub import hf_hub_download

CAMINHO_MODELO = Path("artifacts/jobmatch_ai_artifacts.pkl.gz")
HF_REPO_ID = os.getenv("HF_REPO_ID", "Fe62/jobmatch-ai-artifacts")
HF_FILENAME = os.getenv("HF_MODEL_FILENAME", "jobmatch_ai_artifacts.pkl.gz")


def resolver_caminho_modelo(caminho_modelo: Path = CAMINHO_MODELO) -> Path:
    if caminho_modelo.exists():
        return caminho_modelo

    caminho_baixado = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type="model",
        token=os.getenv("HF_TOKEN") or None,
    )

    return Path(caminho_baixado)


def carregar_artefatos(caminho_modelo: Path = CAMINHO_MODELO) -> dict[str, Any]:
    caminho_resolvido = resolver_caminho_modelo(caminho_modelo)

    with gzip.open(caminho_resolvido, "rb") as arquivo:
        return cloudpickle.load(arquivo)
