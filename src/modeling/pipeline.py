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

class TextConcat(BaseEstimator, TransformerMixin):
    """Concatena colunas textuais em 'text_concat' com normalização segura."""
    def __init__(self, columns=None):
        self.columns = columns or TEXT_COLS

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
        s = df[self.columns].agg(" ".join, axis=1).apply(normalize_text)
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
        ("concat", TextConcat()),
        ("vectorize", text_union),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
    ])
    return pipe