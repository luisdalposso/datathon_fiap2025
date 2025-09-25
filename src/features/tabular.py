from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_tabular_transformer(cat_cols: List[str]):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
