from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import cloudpickle

CAMINHO_MODELO = Path("artifacts/jobmatch_ai_artifacts.pkl.gz")


def carregar_artefatos(caminho_modelo: Path = CAMINHO_MODELO) -> dict[str, Any]:
    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_modelo}. "
        )

    with gzip.open(caminho_modelo, "rb") as arquivo:
        return cloudpickle.load(arquivo)
