from __future__ import annotations
import numpy as np
import pandas as pd

def dcg(relevances: np.ndarray) -> float:
    # log2 discounts starting at 2
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2))
    return float((relevances * discounts).sum())

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    ndcgs = []
    for g, grp in df.groupby("g"):
        if grp.empty:
            continue
        # sort by score desc
        top = grp.sort_values("s", ascending=False).head(k)
        ideal = grp.sort_values("y", ascending=False).head(k)
        dcg_val = dcg(top["y"].to_numpy())
        idcg_val = dcg(ideal["y"].to_numpy())
        if idcg_val == 0.0:
            continue
        ndcgs.append(dcg_val / idcg_val)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
