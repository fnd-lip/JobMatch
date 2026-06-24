from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EXPANSOES_TEXTO = {
    "nuvem": "cloud aws gcp azure",
    "seguranca": "security cybersecurity devsecops vulnerabilities",
    "vulnerabilidades": "vulnerabilities security trivy horusec",
    "esteira": "ci cd github actions gitlab ci pipeline",
    "esteiras": "ci cd github actions gitlab ci pipeline",
    "infraestrutura como codigo": "infrastructure as code terraform",
    "infraestrutura": "infrastructure cloud devops",
    "monitoramento": "monitoring observability",
    "automacao": "automation scripting bash linux",
    "entrega continua": "continuous delivery ci cd",
    "integracao continua": "continuous integration ci cd",
    "analise de dados": "data analysis analytics",
    "estatistica": "statistics",
    "aprendizado de maquina": "machine learning",
    "api": "api apis rest",
    "apis": "api apis rest",
}


def normalizar_texto(valor: Any) -> str:
    texto = "" if valor is None else str(valor).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(caractere for caractere in texto if not unicodedata.combining(caractere))
    texto = re.sub(r"[^a-z0-9+#.\s-]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def normalizar_curriculo_usuario(valor: str) -> str:
    texto = normalizar_texto(valor)
    extras = []

    for termo, expansao in EXPANSOES_TEXTO.items():
        if termo in texto:
            extras.append(expansao)

    if extras:
        texto = f"{texto} {' '.join(extras)}"

    return texto.strip()


def limitar_valor(valor: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
    valor = float(valor)

    if valor < minimo:
        return minimo

    if valor > maximo:
        return maximo

    return valor


def obter_valor(vaga: pd.Series, coluna: str, padrao: Any = "") -> Any:
    if coluna not in vaga:
        return padrao

    valor = vaga.get(coluna)

    if pd.isna(valor):
        return padrao

    return valor


def montar_texto_vaga(vaga: pd.Series) -> str:
    if "texto_vaga" in vaga and pd.notna(vaga.get("texto_vaga")):
        return str(vaga.get("texto_vaga"))

    partes = [
        obter_valor(vaga, "title"),
        obter_valor(vaga, "description"),
        obter_valor(vaga, "skills_desc"),
        obter_valor(vaga, "formatted_work_type"),
        obter_valor(vaga, "formatted_experience_level"),
    ]

    return " ".join(str(parte) for parte in partes if str(parte).strip())


def montar_texto_titulo(vaga: pd.Series) -> str:
    if "texto_titulo" in vaga and pd.notna(vaga.get("texto_titulo")):
        return str(vaga.get("texto_titulo"))

    return str(obter_valor(vaga, "title"))


def montar_entrada_modelo_classificacao(texto_curriculo: str, texto_vaga: str) -> pd.DataFrame:
    curriculo_normalizado = normalizar_curriculo_usuario(texto_curriculo)
    vaga_normalizada = normalizar_texto(texto_vaga)

    texto_par = "curriculo " + curriculo_normalizado + " vaga " + vaga_normalizada

    return pd.DataFrame(
        [
            {
                "texto_par": texto_par,
                "texto_curriculo": curriculo_normalizado,
                "texto_vaga": vaga_normalizada,
            }
        ]
    )


def obter_nomes_skills(artefatos: dict[str, Any]) -> list[str]:
    vetorizador_skills = artefatos["vetorizador_skills"]

    if hasattr(vetorizador_skills, "get_feature_names_out"):
        return list(vetorizador_skills.get_feature_names_out())

    return list(artefatos["nomes_skills"])


def extrair_skills(texto: str, artefatos: dict[str, Any]) -> set[str]:
    vetorizador_skills = artefatos["vetorizador_skills"]
    nomes_skills = obter_nomes_skills(artefatos)

    vetor = vetorizador_skills.transform([normalizar_texto(texto)])
    indices = vetor.nonzero()[1]

    return {nomes_skills[indice] for indice in indices if indice < len(nomes_skills)}


def formatar_skills(skills: set[str], artefatos: dict[str, Any]) -> list[str]:
    mapa_skills = artefatos.get("skill_original_por_normalizada") or {}

    return sorted({mapa_skills.get(skill, skill) for skill in skills})


def calcular_cobertura_skills(skills_compativeis: set[str], skills_vaga: set[str]) -> float:
    if not skills_vaga:
        return 0.0

    return len(skills_compativeis) / len(skills_vaga)


def calcular_score_fit(
    score_aderencia: float,
    score_titulo: float,
    cobertura_skills: float,
    probabilidade_modelo: float | None,
    artefatos: dict[str, Any],
) -> float:
    referencia_score_texto = float(artefatos["referencia_score_texto"])
    referencia_score_titulo = float(artefatos["referencia_score_titulo"])

    score_textual_normalizado = limitar_valor(score_aderencia / referencia_score_texto)
    score_titulo_normalizado = limitar_valor(score_titulo / referencia_score_titulo)
    cobertura_skills = limitar_valor(cobertura_skills)

    if probabilidade_modelo is None:
        return float(
            0.50 * score_textual_normalizado
            + 0.30 * score_titulo_normalizado
            + 0.20 * cobertura_skills
        )

    probabilidade_modelo = limitar_valor(probabilidade_modelo)

    return float(
        0.45 * score_textual_normalizado
        + 0.25 * score_titulo_normalizado
        + 0.20 * cobertura_skills
        + 0.10 * probabilidade_modelo
    )


def classificar_nivel_aderencia(score_fit: float) -> str:
    if score_fit >= 0.70:
        return "Fit forte"

    if score_fit >= 0.50:
        return "Fit potencial"

    return "No Fit"


def obter_probabilidade_modelo(
    texto_curriculo: str,
    texto_vaga: str,
    artefatos: dict[str, Any],
) -> float | None:
    modelo_classificacao = artefatos.get("modelo_classificacao_fit")

    if modelo_classificacao is None:
        return None

    entrada_modelo = montar_entrada_modelo_classificacao(
        texto_curriculo=texto_curriculo,
        texto_vaga=texto_vaga,
    )

    texto_par = entrada_modelo["texto_par"].fillna("").astype(str)

    return float(modelo_classificacao.predict_proba(texto_par)[0][1])
def estimar_salario_anual(
    texto_vaga: str,
    vaga: pd.Series,
    artefatos: dict[str, Any],
) -> float | None:
    salario_alvo = obter_valor(vaga, "salario_alvo", None)

    if salario_alvo is not None:
        return float(salario_alvo)

    salario_padrao = artefatos.get("salario_padrao")

    if salario_padrao is not None:
        return float(salario_padrao)

    return None
def gerar_sugestao_desenvolvimento(skills_faltantes: list[str]) -> str:
    if not skills_faltantes:
        return "Perfil alinhado aos principais requisitos identificados."

    principais_skills = ", ".join(skills_faltantes[:5])

    return f"Priorizar desenvolvimento em: {principais_skills}."


def recomendar_top_vagas_modelo(
    texto_curriculo: str,
    artefatos: dict[str, Any],
    quantidade: int = 5,
    quantidade_candidatas: int = 150,
) -> list[dict[str, Any]]:
    dados_vagas = artefatos["dados_vagas"]
    vetorizador_ranking = artefatos["vetorizador_ranking_vagas"]
    matriz_vagas = artefatos["matriz_vagas_ranking"]
    vetorizador_titulos = artefatos["vetorizador_titulos_vagas"]
    matriz_titulos = artefatos["matriz_titulos_vagas"]

    curriculo_normalizado = normalizar_curriculo_usuario(texto_curriculo)

    vetor_curriculo = vetorizador_ranking.transform([curriculo_normalizado])
    scores_aderencia = cosine_similarity(vetor_curriculo, matriz_vagas).ravel()

    vetor_titulo = vetorizador_titulos.transform([curriculo_normalizado])
    scores_titulo = cosine_similarity(vetor_titulo, matriz_titulos).ravel()

    scores_preliminares = scores_aderencia + (0.25 * scores_titulo)
    indices_candidatos = scores_preliminares.argsort()[::-1][:quantidade_candidatas]

    skills_curriculo = extrair_skills(curriculo_normalizado, artefatos)
    limiar_fit = float(artefatos["limiar_fit"])

    resultados = []

    for indice in indices_candidatos:
        vaga = dados_vagas.iloc[int(indice)]
        texto_vaga = montar_texto_vaga(vaga)

        texto_vaga_normalizado = normalizar_texto(texto_vaga)

        skills_vaga = extrair_skills(texto_vaga_normalizado, artefatos)
        skills_compativeis = skills_curriculo.intersection(skills_vaga)
        skills_faltantes = skills_vaga.difference(skills_curriculo)

        cobertura_skills = calcular_cobertura_skills(skills_compativeis, skills_vaga)

        probabilidade_modelo = obter_probabilidade_modelo(
            texto_curriculo=curriculo_normalizado,
            texto_vaga=texto_vaga_normalizado,
            artefatos=artefatos,
        )

        score_fit = calcular_score_fit(
            score_aderencia=float(scores_aderencia[indice]),
            score_titulo=float(scores_titulo[indice]),
            cobertura_skills=cobertura_skills,
            probabilidade_modelo=probabilidade_modelo,
            artefatos=artefatos,
        )

        classificacao = "Fit" if score_fit >= limiar_fit else "No Fit"
        salario_anual = estimar_salario_anual(texto_vaga_normalizado, vaga, artefatos)

        skills_compativeis_lista = formatar_skills(skills_compativeis, artefatos)
        skills_faltantes_lista = formatar_skills(skills_faltantes, artefatos)

        resultados.append(
            {
                "job_id": obter_valor(vaga, "job_id", int(indice)),
                "titulo_vaga": obter_valor(vaga, "title", "Vaga sem título"),
                "empresa": obter_valor(vaga, "company_name", "Empresa não informada"),
                "localizacao": obter_valor(vaga, "location", "Localização não informada"),
                "score_aderencia": round(float(scores_aderencia[indice]), 4),
                "score_titulo": round(float(scores_titulo[indice]), 4),
                "cobertura_skills": round(cobertura_skills, 4),
                "probabilidade_modelo_fit": (
                    round(probabilidade_modelo, 4)
                    if probabilidade_modelo is not None
                    else None
                ),
                "score_fit": round(score_fit, 4),
                "classificacao": classificacao,
                "nivel_aderencia": classificar_nivel_aderencia(score_fit),
                "skills_compativeis": skills_compativeis_lista[:10],
                "skills_faltantes": skills_faltantes_lista[:10],
                "sugestao_desenvolvimento": gerar_sugestao_desenvolvimento(
                    skills_faltantes_lista
                ),
                "salario_estimado_anual": round(salario_anual, 2)
                if salario_anual is not None
                else None,
                "moeda": obter_valor(vaga, "currency", "USD"),
            }
        )

    resultados = sorted(
        resultados,
        key=lambda item: item["score_fit"],
        reverse=True,
    )

    return resultados[:quantidade]
