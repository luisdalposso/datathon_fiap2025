# src/modeling/pipeline.py
from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from ..features.text_clean import normalize_text
from ..features.nlp_tfidf import build_tfidf_vectorizer, build_char_vectorizer

TEXT_COLS = [
    "cv_pt",
    "principais_atividades",
    "competencias",
    "observacoes",
    "titulo_vaga",
]

JOB_COLS = ["principais_atividades", "competencias", "observacoes", "titulo_vaga"]
CV_COL = "cv_pt"

class TextConcat(BaseEstimator, TransformerMixin):
    """Concatena colunas textuais em 'text_concat' com normalização segura.
       Dá mais peso ao texto da vaga (job_weight) para reforçar o match vaga↔candidato.
    """
    def __init__(self, columns=None, job_weight: int = 2, cv_weight: int = 1):
        self.columns = columns or TEXT_COLS
        self.job_weight = max(1, int(job_weight))
        self.cv_weight = max(1, int(cv_weight))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X.copy()

        # garante presença, str e NaN -> ""
        for col in self.columns:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        # separa cv e texto da vaga
        cv_txt = df[CV_COL]
        job_txt = df[JOB_COLS].agg(" ".join, axis=1)

        # monta com pesos (vaga repetida job_weight vezes)
        parts = []
        for _ in range(self.cv_weight):
            parts.append(cv_txt)
        for _ in range(self.job_weight):
            parts.append(job_txt)

        s = parts[0]
        for p in parts[1:]:
            s = s.str.cat(p, sep=" ")

        # normaliza
        s = s.apply(normalize_text)

        # retorna DataFrame com nome fixo (evita erros de feature_names_out)
        return pd.DataFrame({"text_concat": s})

def build_pipeline() -> Pipeline:
    text_union = ColumnTransformer(
        transformers=[
            ("tfidf_word", build_tfidf_vectorizer(), "text_concat"),
            ("tfidf_char", build_char_vectorizer(), "text_concat"),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    pipe = Pipeline(steps=[
        ("concat", TextConcat(job_weight=2, cv_weight=1)),  # vaga com peso 2×
        ("vectorize", text_union),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
    ])
    return pipe