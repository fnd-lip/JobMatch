from __future__ import annotations

from typing import Any

import streamlit as st

from src.pipeline.artefatos import carregar_artefatos
from src.pipeline.recomendador_modelo import recomendar_top_vagas_modelo

EXEMPLOS_PERFIL = {
    "Dados / Machine Learning": (
        "Tenho experiência com Python, SQL, pandas, scikit-learn, machine learning, "
        "análise de dados, estatística, dashboards, APIs, Docker e FastAPI."
    ),
    "Backend": (
        "Tenho experiência com Python, FastAPI, SQL, APIs REST, Docker, testes "
        "automatizados e integração de sistemas."
    ),
    "DevSecOps / Cloud": (
        "Tenho experiência em Infraestrutura e DevSecOps, provisionamento em nuvem "
        "AWS e GCP, CI/CD com GitHub Actions e GitLab CI, Docker, Bash, Linux, "
        "Terraform, Trivy, Horusec e monitoramento de aplicações."
    ),
}


def formatar_moeda(valor: float | None, moeda: str = "USD") -> str:
    if valor is None:
        return "Não informado"

    valor_formatado = f"{valor:,.2f}"
    valor_formatado = (
        valor_formatado.replace(",", "X").replace(".", ",").replace("X", ".")
    )

    if moeda == "BRL":
        return f"R\\$ {valor_formatado}"

    if moeda == "USD":
        return f"US\\$ {valor_formatado}"

    return f"{moeda} {valor_formatado}"


def formatar_lista_skills(skills: list[str]) -> str:
    if not skills:
        return "Nenhuma"

    return ", ".join(skills)


def obter_icone_classificacao(classificacao: str) -> str:
    if classificacao == "Fit":
        return "✅"

    return "⚠️"


def exibir_card_vaga(posicao: int, vaga: dict[str, Any]) -> None:
    classificacao = vaga["classificacao"]
    icone = obter_icone_classificacao(classificacao)
    salario_anual = vaga["salario_estimado_anual"]
    moeda = vaga["moeda"]

    with st.container(border=True):
        coluna_titulo, coluna_score = st.columns([0.70, 0.30])

        with coluna_titulo:
            st.subheader(f"{posicao}. {vaga['titulo_vaga']}")
            st.caption(f"{vaga['empresa']} | {vaga['localizacao']}")
            st.caption(f"{icone} {classificacao} — {vaga['nivel_aderencia']}")

        with coluna_score:
            st.metric("Score Fit", f"{vaga['score_fit']:.4f}")

        coluna_1, coluna_2, coluna_3 = st.columns(3)
        coluna_1.metric("Score textual", f"{vaga['score_aderencia']:.4f}")
        coluna_2.metric("Score título", f"{vaga['score_titulo']:.4f}")
        probabilidade_modelo = vaga["probabilidade_modelo_fit"]
        coluna_3.metric(
            "Prob. modelo",
            "N/A" if probabilidade_modelo is None else f"{probabilidade_modelo:.4f}",
        )

        st.markdown("**Skills compatíveis**")
        st.success(formatar_lista_skills(vaga["skills_compativeis"]))

        st.markdown("**Skills faltantes**")
        if vaga["skills_faltantes"]:
            st.warning(formatar_lista_skills(vaga["skills_faltantes"]))
        else:
            st.info("Nenhuma skill faltante identificada.")

        st.markdown("**Sugestão de desenvolvimento**")
        st.write(vaga["sugestao_desenvolvimento"])

        st.markdown("**Salário anual estimado**")
        if salario_anual is not None:
            salario_min = salario_anual * 0.85
            salario_max = salario_anual * 1.15

            st.write(
                f"{formatar_moeda(salario_min, moeda)} a "
                f"{formatar_moeda(salario_max, moeda)} "
                f"(média {formatar_moeda(salario_anual, moeda)})"
            )
        else:
            st.write("Não informado")


@st.cache_resource
def carregar_modelo() -> dict[str, Any]:
    return carregar_artefatos()


def main() -> None:
    st.set_page_config(
        page_title="JobMatch AI",
        page_icon="💼",
        layout="wide",
    )

    st.title("💼 JobMatch AI")
    st.caption("Recomendação de vagas com o artefato completo de Machine Learning.")

    artefatos = carregar_modelo()
    total_vagas = len(artefatos["dados_vagas"])

    with st.sidebar:
        st.header("Configurações")

        top_k = st.slider(
            "Quantidade de vagas",
            min_value=1,
            max_value=10,
            value=5,
        )

        exemplo_escolhido = st.selectbox(
            "Exemplo de perfil",
            options=["Personalizado", *EXEMPLOS_PERFIL.keys()],
        )

        st.divider()

        st.metric("Vagas no modelo", f"{total_vagas:,}".replace(",", "."))
        st.write("A recomendação usa o artefato `jobmatch_ai_artifacts.pkl.gz`.")

    perfil_padrao = ""

    if exemplo_escolhido != "Personalizado":
        perfil_padrao = EXEMPLOS_PERFIL[exemplo_escolhido]

    st.markdown("""
        Informe um perfil profissional ou cole um currículo resumido.  
        O sistema compara o perfil com as vagas do artefato treinado e retorna ranking,
        score, classificação Fit/No Fit, skills e salário estimado.
        """)

    candidate_profile = st.text_area(
        "Perfil do candidato",
        value=perfil_padrao,
        height=220,
        placeholder=(
            "Exemplo: Tenho experiência com Python, SQL, machine learning, "
            "Docker, cloud, APIs e análise de dados."
        ),
    )

    botao_buscar = st.button("Encontrar vagas compatíveis", type="primary")

    if botao_buscar:
        if not candidate_profile.strip():
            st.warning("Informe o perfil do candidato antes de buscar vagas.")
            st.stop()

        with st.spinner("Analisando perfil com o modelo completo..."):
            recommendations = recomendar_top_vagas_modelo(
                texto_curriculo=candidate_profile,
                artefatos=artefatos,
                quantidade=top_k,
                quantidade_candidatas=30,
            )

        total_fit = sum(1 for vaga in recommendations if vaga["classificacao"] == "Fit")
        melhor_score = recommendations[0]["score_fit"] if recommendations else 0

        st.divider()

        coluna_1, coluna_2, coluna_3 = st.columns(3)

        coluna_1.metric("Vagas analisadas", f"{total_vagas:,}".replace(",", "."))
        coluna_2.metric("Recomendações Fit", total_fit)
        coluna_3.metric("Melhor score", f"{melhor_score:.4f}")

        st.subheader("Top vagas recomendadas")

        for index, job in enumerate(recommendations, start=1):
            exibir_card_vaga(index, job)


if __name__ == "__main__":
    main()
