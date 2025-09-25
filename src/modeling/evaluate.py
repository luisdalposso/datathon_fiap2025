from __future__ import annotations
import numpy as np
import pandas as pd

def dcg(relevances: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2))
    return float((relevances * discounts).sum())

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    ndcgs = []
    for _, grp in df.groupby("g", sort=False):
        if grp.empty:
            continue
        top = grp.sort_values("s", ascending=False).head(k)
        ideal = grp.sort_values("y", ascending=False).head(k)
        dcg_val = dcg(top["y"].to_numpy())
        idcg_val = dcg(ideal["y"].to_numpy())
        if idcg_val == 0.0:
            continue
        ndcgs.append(dcg_val / idcg_val)
    return float(np.mean(ndcgs)) if ndcgs else 0.0

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    vals = []
    for _, grp in df.groupby("g", sort=False):
        if grp.empty:
            continue
        topk = grp.sort_values("s", ascending=False).head(k)
        vals.append(float(topk["y"].sum()) / float(len(topk)) if len(topk) > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    vals = []
    for _, grp in df.groupby("g", sort=False):
        total_pos = int(grp["y"].sum())
        if total_pos == 0:
            # pula grupos sem relevantes (recall indefinido)
            continue
        topk = grp.sort_values("s", ascending=False).head(k)
        got_pos = int(topk["y"].sum())
        vals.append(got_pos / float(total_pos))
    return float(np.mean(vals)) if vals else 0.0

def mrr(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    rrs = []
    for _, grp in df.groupby("g", sort=False):
        if grp["y"].sum() == 0:
            continue
        ordered = grp.sort_values("s", ascending=False).reset_index(drop=True)
        # posição (1-indexed) do primeiro relevante
        pos = ordered.index[ordered["y"] == 1]
        if len(pos) == 0:
            continue
        rank = int(pos[0]) + 1
        rrs.append(1.0 / rank)
    return float(np.mean(rrs)) if rrs else 0.0