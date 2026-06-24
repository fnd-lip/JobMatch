from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv

from src.pipeline.inferencia.artefatos import CAMINHO_MODELO

load_dotenv(override=True)


def configurar_mlflow() -> str:
    dono_repositorio = os.getenv("DAGSHUB_REPO_OWNER")
    nome_repositorio = os.getenv("DAGSHUB_REPO_NAME")

    if not dono_repositorio or not nome_repositorio:
        raise ValueError(
            "Defina DAGSHUB_REPO_OWNER e DAGSHUB_REPO_NAME no arquivo .env."
        )

    tracking_uri = f"https://dagshub.com/{dono_repositorio}/{nome_repositorio}.mlflow"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("jobmatch-ai-validacao")

    return tracking_uri


def registrar_metrica(nome: str, valor: Any) -> None:
    if valor is None:
        return

    mlflow.log_metric(nome, float(valor))


def registrar_execucao_mlflow(
    artefatos: dict[str, Any],
    inconsistencias: list[str],
    caminho_relatorio: Path,
) -> None:
    tracking_uri = configurar_mlflow()

    dados_vagas = artefatos.get("dados_vagas")
    metricas_classificacao = artefatos.get("metricas_classificacao") or {}
    metricas_salario = artefatos.get("metricas_salario") or {}

    with mlflow.start_run(run_name="jobmatch_ai_validacao_artefato"):
        mlflow.log_param("tracking_uri", tracking_uri)
        mlflow.log_param("arquivo_modelo", str(CAMINHO_MODELO))
        mlflow.log_param("limiar_fit", artefatos.get("limiar_fit"))
        mlflow.log_param("referencia_score_texto", artefatos.get("referencia_score_texto"))
        mlflow.log_param("referencia_score_titulo", artefatos.get("referencia_score_titulo"))

        if dados_vagas is not None:
            registrar_metrica("qtd_vagas", len(dados_vagas))

        registrar_metrica("classificacao_acuracia", metricas_classificacao.get("acuracia"))
        registrar_metrica("classificacao_f1_macro", metricas_classificacao.get("f1_macro"))

        registrar_metrica("salario_mae", metricas_salario.get("mae"))
        registrar_metrica("salario_rmse", metricas_salario.get("rmse"))
        registrar_metrica("salario_r2", metricas_salario.get("r2"))

        registrar_metrica("qtd_inconsistencias", len(inconsistencias))

        mlflow.log_artifact(str(CAMINHO_MODELO), artifact_path="modelo")
        mlflow.log_artifact(str(caminho_relatorio), artifact_path="validacao")
