from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.utils.text import normalize_text

FIT_THRESHOLD = 0.25


def recommend_jobs(
    candidate_profile: str,
    jobs: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not jobs:
        return []

    candidate_text = normalize_text(candidate_profile)
    job_texts = [_build_job_text(job) for job in jobs]
    all_texts = [candidate_text, *job_texts]

    if not any(text.strip() for text in all_texts):
        return [_format_result(job, 0.0, candidate_text) for job in jobs[:top_k]]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_indexes = np.argsort(scores)[::-1][:top_k]

    return [
        _format_result(jobs[int(index)], float(scores[int(index)]), candidate_text)
        for index in ranked_indexes
    ]


def _build_job_text(job: dict[str, Any]) -> str:
    title = job.get("title", "")
    description = job.get("description", "")
    skills = job.get("skills", [])

    if isinstance(skills, list):
        skills_text = " ".join(str(skill) for skill in skills)
    else:
        skills_text = str(skills)

    return normalize_text(f"{title} {description} {skills_text}")


def _format_result(
    job: dict[str, Any],
    score: float,
    candidate_text: str,
) -> dict[str, Any]:
    matched_skills, missing_skills = _compare_skills(candidate_text, job.get("skills", []))

    return {
        "job_id": str(job.get("job_id", "")),
        "title": str(job.get("title", "")),
        "score": round(score, 4),
        "classification": "Fit" if score >= FIT_THRESHOLD else "No Fit",
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "salary_estimate": _estimate_salary(job),
    }


def _compare_skills(candidate_text: str, job_skills: Any) -> tuple[list[str], list[str]]:
    if not isinstance(job_skills, list):
        return [], []

    matched = []
    missing = []

    for skill in job_skills:
        skill_name = str(skill).strip()
        normalized_skill = normalize_text(skill_name)

        if not normalized_skill:
            continue

        if normalized_skill in candidate_text:
            matched.append(skill_name)
        else:
            missing.append(skill_name)

    return matched, missing


def _estimate_salary(job: dict[str, Any]) -> dict[str, float | None]:
    salary_min = job.get("salary_min")
    salary_max = job.get("salary_max")

    if salary_min is None or salary_max is None:
        return {"min": None, "max": None, "average": None}

    salary_min = float(salary_min)
    salary_max = float(salary_max)

    return {
        "min": salary_min,
        "max": salary_max,
        "average": round((salary_min + salary_max) / 2, 2),
    }
