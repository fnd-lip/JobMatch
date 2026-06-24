from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_jobs_from_csv(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    dataframe = pd.read_csv(path, encoding="utf-8-sig")

    required_columns = {
        "job_id",
        "title",
        "description",
        "skills",
        "salary_min",
        "salary_max",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")

    jobs: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        jobs.append(
            {
                "job_id": str(row["job_id"]),
                "title": str(row["title"]),
                "description": str(row["description"]),
                "skills": _parse_skills(row["skills"]),
                "salary_min": float(row["salary_min"]),
                "salary_max": float(row["salary_max"]),
            }
        )

    return jobs


def _parse_skills(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    return [skill.strip() for skill in str(value).split(";") if skill.strip()]
