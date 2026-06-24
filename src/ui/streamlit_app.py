from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.pipeline.data import load_jobs_from_csv
from src.pipeline.recommender import recommend_jobs

st.set_page_config(
    page_title="JobMatch AI",
    page_icon="💼",
    layout="centered",
)

st.title("💼 JobMatch AI")
st.caption("Recomendação inteligente de vagas usando TF-IDF e similaridade por cosseno.")

st.markdown(
    """
    Descreva seu perfil profissional ou cole um currículo resumido.
    O sistema retorna as vagas mais aderentes, score de compatibilidade,
    classificação Fit/No Fit, skills compatíveis, skills faltantes e faixa salarial estimada.
    """
)

candidate_profile = st.text_area(
    "Perfil do candidato",
    height=180,
    placeholder=(
        "Exemplo: Tenho experiência com Python, SQL, pandas, machine learning, "
        "APIs, Docker e análise de dados."
    ),
)

top_k = st.slider(
    "Quantidade de vagas recomendadas",
    min_value=1,
    max_value=5,
    value=5,
)

if st.button("Encontrar vagas"):
    if not candidate_profile.strip():
        st.warning("Informe o perfil do candidato antes de buscar vagas.")
        st.stop()

    jobs = load_jobs_from_csv(Path("data/samples/jobs_sample.csv"))
    recommendations = recommend_jobs(candidate_profile, jobs, top_k=top_k)

    st.subheader("Top vagas recomendadas")

    for index, job in enumerate(recommendations, start=1):
        salary = job["salary_estimate"]

        with st.expander(f"{index}. {job['title']} — {job['classification']}", expanded=True):
            st.write(f"**Score de aderência:** {job['score']}")
            st.write(f"**Classificação:** {job['classification']}")
            st.write(f"**Skills compatíveis:** {', '.join(job['matched_skills']) or 'Nenhuma'}")
            st.write(f"**Skills faltantes:** {', '.join(job['missing_skills']) or 'Nenhuma'}")

            if salary["average"] is not None:
                st.write(
                    "**Faixa salarial estimada:** "
                    f"R$ {salary['min']:.2f} a R$ {salary['max']:.2f} "
                    f"(média R$ {salary['average']:.2f})"
                )
