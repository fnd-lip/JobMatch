from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CAMINHO_RELATORIO = Path("artifacts/relatorio_validacao_modelo.json")

CHAVES_OBRIGATORIAS = [
    "dados_vagas",
    "modelo_classificacao_fit",
    "modelo_estimativa_salario",
    "vetorizador_ranking_vagas",
    "matriz_vagas_ranking",
    "vetorizador_titulos_vagas",
    "matriz_titulos_vagas",
    "vetorizador_skills",
    "nomes_skills",
    "skill_original_por_normalizada",
    "salario_padrao",
    "referencia_score_texto",
    "referencia_score_titulo",
    "limiar_fit",
]


def validar_artefatos(artefatos: dict[str, Any]) -> list[str]:
    inconsistencias = []

    for chave in CHAVES_OBRIGATORIAS:
        if chave not in artefatos:
            inconsistencias.append(f"Chave ausente: {chave}")

    if inconsistencias:
        return inconsistencias

    dados_vagas = artefatos["dados_vagas"]
    matriz_vagas = artefatos["matriz_vagas_ranking"]
    matriz_titulos = artefatos["matriz_titulos_vagas"]
    modelo_classificacao = artefatos["modelo_classificacao_fit"]
    modelo_salario = artefatos["modelo_estimativa_salario"]

    if len(dados_vagas) == 0:
        inconsistencias.append("dados_vagas está vazio.")

    if matriz_vagas.shape[0] != len(dados_vagas):
        inconsistencias.append("matriz_vagas_ranking não bate com dados_vagas.")

    if matriz_titulos.shape[0] != len(dados_vagas):
        inconsistencias.append("matriz_titulos_vagas não bate com dados_vagas.")

    if modelo_classificacao is None:
        inconsistencias.append("modelo_classificacao_fit está vazio.")

    if modelo_classificacao is not None and not hasattr(modelo_classificacao, "predict_proba"):
        inconsistencias.append("modelo_classificacao_fit não possui predict_proba.")

    if modelo_salario is None:
        inconsistencias.append("modelo_estimativa_salario está vazio.")

    if not 0 <= float(artefatos["limiar_fit"]) <= 1:
        inconsistencias.append("limiar_fit está fora do intervalo entre 0 e 1.")

    if float(artefatos["referencia_score_texto"]) <= 0:
        inconsistencias.append("referencia_score_texto deve ser maior que zero.")

    if float(artefatos["referencia_score_titulo"]) <= 0:
        inconsistencias.append("referencia_score_titulo deve ser maior que zero.")

    return inconsistencias


def salvar_relatorio_validacao(
    artefatos: dict[str, Any],
    inconsistencias: list[str],
    caminho_relatorio: Path = CAMINHO_RELATORIO,
) -> Path:
    metricas_classificacao = artefatos.get("metricas_classificacao") or {}
    metricas_salario = artefatos.get("metricas_salario") or {}

    relatorio = {
        "status": "ok" if not inconsistencias else "com_inconsistencias",
        "qtd_inconsistencias": len(inconsistencias),
        "inconsistencias": inconsistencias,
        "metricas_classificacao": {
            "acuracia": metricas_classificacao.get("acuracia"),
            "f1_macro": metricas_classificacao.get("f1_macro"),
            "limiar": metricas_classificacao.get("limiar"),
        },
        "metricas_salario": metricas_salario,
    }

    caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_relatorio, "w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2, default=str)

    return caminho_relatorio
