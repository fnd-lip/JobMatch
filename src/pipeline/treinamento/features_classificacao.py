from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_STATE = 42


def criar_pipeline_classificacao_estavel() -> Pipeline:
    extrator_features = FeatureUnion(
        [
            (
                "tfidf_palavras",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 3),
                    min_df=2,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "tfidf_caracteres",
                TfidfVectorizer(
                    max_features=15000,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classificador = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("features", extrator_features),
            ("classificador", classificador),
        ]
    )
