from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.pipeline.inferencia.artefatos import CAMINHO_MODELO
from src.pipeline.treinamento.features_classificacao import (
    RANDOM_STATE,
    criar_pipeline_classificacao_estavel,
)

CAMINHO_DADOS_CLASSIFICACAO = Path("data/processed/dados_classificacao_fit.csv")


def carregar_dados_classificacao() -> tuple[pd.Series, pd.Series]:
    dados = pd.read_csv(CAMINHO_DADOS_CLASSIFICACAO, encoding="utf-8")

    dados = dados.dropna(subset=["texto_par", "rotulo_fit"]).copy()
    dados["rotulo_fit"] = dados["rotulo_fit"].astype(int)

    textos = dados["texto_par"].fillna("").astype(str)
    rotulos = dados["rotulo_fit"]

    return textos, rotulos


def carregar_artefatos() -> dict[str, Any]:
    with gzip.open(CAMINHO_MODELO, "rb") as arquivo:
        return cloudpickle.load(arquivo)


def salvar_artefatos(artefatos: dict[str, Any]) -> None:
    with gzip.open(CAMINHO_MODELO, "wb") as arquivo:
        cloudpickle.dump(artefatos, arquivo)


def escolher_limiar(
    probabilidades: np.ndarray,
    rotulos_teste: pd.Series,
) -> tuple[float, float, float, np.ndarray]:
    melhor_limiar = 0.5
    melhor_acuracia = 0.0
    melhor_f1 = 0.0
    melhores_previsoes = (probabilidades >= melhor_limiar).astype(int)

    for limiar in np.arange(0.30, 0.71, 0.01):
        previsoes = (probabilidades >= limiar).astype(int)
        acuracia = accuracy_score(rotulos_teste, previsoes)
        f1_macro = f1_score(rotulos_teste, previsoes, average="macro")

        if f1_macro > melhor_f1:
            melhor_limiar = round(float(limiar), 2)
            melhor_acuracia = float(acuracia)
            melhor_f1 = float(f1_macro)
            melhores_previsoes = previsoes

    return melhor_limiar, melhor_acuracia, melhor_f1, melhores_previsoes


def main() -> None:
    textos, rotulos = carregar_dados_classificacao()
    artefatos = carregar_artefatos()

    textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(
        textos,
        rotulos,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=rotulos,
    )

    modelo_validacao = criar_pipeline_classificacao_estavel()

    print("Treinando classificador Fit/No Fit estável...")
    modelo_validacao.fit(textos_treino, rotulos_treino)

    probabilidades = modelo_validacao.predict_proba(textos_teste)[:, 1]

    limiar, acuracia, f1_macro, previsoes = escolher_limiar(
        probabilidades=probabilidades,
        rotulos_teste=rotulos_teste,
    )

    matriz = confusion_matrix(rotulos_teste, previsoes)

    print("Treinando modelo final com todos os dados...")
    modelo_final = criar_pipeline_classificacao_estavel()
    modelo_final.fit(textos, rotulos)

    artefatos["modelo_classificacao_fit"] = modelo_final
    artefatos["limiar_fit"] = limiar
    artefatos["metricas_classificacao"] = {
        "modelo": "tfidf_logistic_regression_estavel_src",
        "limiar": limiar,
        "acuracia": acuracia,
        "f1_macro": f1_macro,
        "matriz_confusao": matriz.tolist(),
    }

    salvar_artefatos(artefatos)

    print("Classificador atualizado no artefato atual.")
    print("Limiar:", limiar)
    print("Acurácia:", acuracia)
    print("F1 macro:", f1_macro)
    print("Matriz de confusão:")
    print(matriz)


if __name__ == "__main__":
    main()
