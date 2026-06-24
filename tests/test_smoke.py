from pathlib import Path

from src.pipeline.data import load_jobs_from_csv
from src.pipeline.recommender import recommend_jobs
from src.pipeline.text import normalize_text


def test_smoke_jobmatch_pipeline():
    jobs = load_jobs_from_csv(Path("data/samples/jobs_sample.csv"))

    candidate_profile = """
    Desenvolvedor Python com SQL, pandas, machine learning,
    APIs, Docker e análise de dados.
    """

    recommendations = recommend_jobs(candidate_profile, jobs, top_k=5)

    assert len(jobs) >= 5
    assert len(recommendations) == 5

    first = recommendations[0]

    assert "job_id" in first
    assert "title" in first
    assert "score" in first
    assert "classification" in first
    assert "matched_skills" in first
    assert "missing_skills" in first
    assert "salary_estimate" in first

    assert 0 <= first["score"] <= 1
    assert first["classification"] in ["Fit", "No Fit"]


def test_normalize_text_smoke():
    assert normalize_text(" Análise de Dados ") == "analise de dados"
