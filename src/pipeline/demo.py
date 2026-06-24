from __future__ import annotations

from pathlib import Path

from src.pipeline.data import load_jobs_from_csv
from src.pipeline.recommender import recommend_jobs


def main() -> None:
    candidate_profile = """
    Desenvolvedor com experiência em Python, SQL, pandas,
    machine learning, APIs, Docker e análise de dados.
    """

    jobs = load_jobs_from_csv(Path("data/samples/jobs_sample.csv"))
    recommendations = recommend_jobs(candidate_profile, jobs, top_k=5)

    print("\nJobMatch AI - Top 5 vagas recomendadas\n")

    for index, job in enumerate(recommendations, start=1):
        salary = job["salary_estimate"]

        print(f"{index}. {job['title']}")
        print(f"   Score: {job['score']}")
        print(f"   Classificação: {job['classification']}")
        print(f"   Skills compatíveis: {', '.join(job['matched_skills']) or 'Nenhuma'}")
        print(f"   Skills faltantes: {', '.join(job['missing_skills']) or 'Nenhuma'}")

        if salary["average"] is not None:
            print(
                f"   Salário estimado: R$ {salary['min']:.2f} "
                f"a R$ {salary['max']:.2f} "
                f"(média R$ {salary['average']:.2f})"
            )

        print()


if __name__ == "__main__":
    main()
