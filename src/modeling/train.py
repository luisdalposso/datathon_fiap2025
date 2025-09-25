from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.dummy import DummyClassifier
import numpy as np

from ..config.settings import (
    APPLICANTS_PATH, VAGAS_PATH, PROSPECTS_PATH,
    MODELS_DIR, REPORTS_DIR, RANDOM_STATE
)
from ..data.loaders import load_applicants, load_jobs, load_prospects
from ..labeling.targets import map_status_to_label
from .pipeline import build_pipeline
from .evaluate import ndcg_at_k


def make_training_table(app_df: pd.DataFrame, job_df: pd.DataFrame, prs_df: pd.DataFrame) -> pd.DataFrame:
    # Join prospects with applicants and jobs
    df = prs_df.copy()
    df["y"] = df["situacao"].map(map_status_to_label).fillna(0).astype(int)

    df = df.merge(app_df, on="applicant_id", how="left", validate="m:1")
    df = df.merge(job_df, on="job_id", how="left", validate="m:1")

    # Garante presença das colunas textuais, mesmo se faltarem após o merge
    text_cols = ["cv_pt", "principais_atividades", "competencias", "observacoes", "titulo_vaga"]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""

    # drop rows sem qualquer texto útil
    has_any_text = (
        df["cv_pt"].fillna("").astype(str).str.len()
        + df["principais_atividades"].fillna("").astype(str).str.len()
        + df["competencias"].fillna("").astype(str).str.len()
        + df["observacoes"].fillna("").astype(str).str.len()
        + df["titulo_vaga"].fillna("").astype(str).str.len()
    ) > 0
    df = df[has_any_text]

    # chaves obrigatórias
    df = df.dropna(subset=["job_id", "applicant_id"]).reset_index(drop=True)
    return df


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    app_df = load_applicants(APPLICANTS_PATH)
    job_df = load_jobs(VAGAS_PATH)
    # sanitize jobs to avoid merge errors
    job_df = job_df[job_df["job_id"].notna() & (job_df["job_id"] != "")]
    job_df = job_df.drop_duplicates("job_id", keep="first")
    prs_df = load_prospects(PROSPECTS_PATH)

    print("jobs total:", len(job_df), "ids únicos:", job_df["job_id"].nunique())
    dups = job_df["job_id"][job_df["job_id"].duplicated()].unique()
    print("duplicatas:", list(dups[:5]))

    data = make_training_table(app_df, job_df, prs_df)
    if data.empty:
        raise RuntimeError(
            "Tabela de treino ficou vazia. Verifique a interseção entre prospects/applicants/vagas "
            "e se há texto disponível para TF-IDF."
        )

    y = data["y"].to_numpy()
    groups = data["job_id"].to_numpy()
    X = data  # a pipeline cuida dos textos

    pipe = build_pipeline()

    # número de grupos e splits seguros (evita splits “apertados”)
    n_groups = int(pd.Series(groups).nunique())
    n_splits = 3 if n_groups >= 6 else 2

    ndcgs, rocs, f1s = [], [], []

    # tenta Estratificado por grupo; se não der, cai para GroupKFold
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    try:
        split_iter = splitter.split(X, y, groups)
    except Exception:
        split_iter = GroupKFold(n_splits=n_splits).split(X, y, groups)

    valid_folds = 0
    for fold, (tr, va) in enumerate(split_iter):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        gva = groups[va]

        # Se o treino tiver uma única classe, pula o fold
        if len(np.unique(ytr)) < 2:
            print(f"[Fold {fold}] pulado: treino com classe única ({np.unique(ytr)})")
            continue

        pipe.fit(Xtr, ytr)

        # obtém score contínuo 0-1
        if hasattr(pipe[-1], "predict_proba"):
            s = pipe.predict_proba(Xva)[:, 1]
        else:
            if hasattr(pipe[-1], "decision_function"):
                dfu = pipe.decision_function(Xva)
                s = (dfu - dfu.min()) / (dfu.max() - dfu.min() + 1e-9)
            else:
                s = pipe.predict(Xva).astype(float)

        nd = ndcg_at_k(y_true=yva, y_score=s, groups=gva, k=5)
        ndcgs.append(nd)

        try:
            rocs.append(roc_auc_score(yva, s))
        except Exception:
            pass
        preds = (s >= 0.5).astype(int)
        f1s.append(f1_score(yva, preds))

        print(f"[Fold {fold}] NDCG@5={nd:.4f} F1={f1s[-1]:.4f}")
        valid_folds += 1

    if valid_folds == 0:
        print("[AVISO] Nenhum fold válido (classe única nos splits). Métricas serão 0.")
        metrics = {"NDCG@5_mean": 0.0, "F1_mean": 0.0, "ROC_AUC_mean": 0.0}
    else:
        metrics = {
            "NDCG@5_mean": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
            "F1_mean": float(sum(f1s) / len(f1s)) if f1s else 0.0,
            "ROC_AUC_mean": float(sum(rocs) / len(rocs)) if rocs else 0.0,
        }
    print("[Metrics]", json.dumps(metrics, ensure_ascii=False, indent=2))

    # =======================
    # Fit final + salvamento
    # =======================
    if len(np.unique(y)) < 2:
        print("[AVISO] Dataset completo com classe única — usando DummyClassifier(most_frequent).")
        pipe.set_params(clf=DummyClassifier(strategy="most_frequent"))
    pipe.fit(X, y)

    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(json.dumps({
        "features": ["text_concat via TF-IDF (word+char)"],
        "target": "y",
        "metrics": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] {MODELS_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()