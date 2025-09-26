from __future__ import annotations
from pathlib import Path
import json
import time
from datetime import datetime
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
from .evaluate import ndcg_at_k, precision_at_k, recall_at_k, mrr

# Alvo de workload por vaga: quantos candidatos o recrutador quer ver no topo
TARGET_K = 5

def _fmt_secs(seconds: float) -> str:
    m, s = divmod(seconds, 60.0)
    h, m = divmod(m, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    t0 = time.perf_counter()
    print(f"[TIMER] Início do treino: { _now() }")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    app_df = load_applicants(APPLICANTS_PATH)
    job_df = load_jobs(VAGAS_PATH)
    prs_df = load_prospects(PROSPECTS_PATH)
    t_load = time.perf_counter() - t_load0
    print(f"[TIMER] Carregamento de dados: { _fmt_secs(t_load) }")

    # sanitize jobs to avoid merge errors
    job_df = job_df[job_df["job_id"].notna() & (job_df["job_id"] != "")]
    job_df = job_df.drop_duplicates("job_id", keep="first")

    print("jobs total:", len(job_df), "ids únicos:", job_df["job_id"].nunique())
    dups = job_df["job_id"][job_df["job_id"].duplicated()].unique()
    print("duplicatas:", list(dups[:5]))

    t_prep0 = time.perf_counter()
    data = make_training_table(app_df, job_df, prs_df)
    t_prep = time.perf_counter() - t_prep0
    print(f"[TIMER] Montagem da tabela de treino: { _fmt_secs(t_prep) } (linhas={len(data)})")

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
    kth_scores: list[float] = []

    # tenta Estratificado por grupo; se não der, cai para GroupKFold
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    try:
        split_iter = splitter.split(X, y, groups)
    except Exception:
        split_iter = GroupKFold(n_splits=n_splits).split(X, y, groups)

    t_cv0 = time.perf_counter()
    valid_folds = 0
    for fold, (tr, va) in enumerate(split_iter):
        t_fold0 = time.perf_counter()

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

        # Métricas de ranking
        nd = ndcg_at_k(y_true=yva, y_score=s, groups=gva, k=TARGET_K)
        p5 = precision_at_k(y_true=yva, y_score=s, groups=gva, k=TARGET_K)
        r5 = recall_at_k(y_true=yva, y_score=s, groups=gva, k=TARGET_K)
        mrrv = mrr(y_true=yva, y_score=s, groups=gva)
        ndcgs.append(nd)

        # binárias de referência
        try:
            rocs.append(roc_auc_score(yva, s))
        except Exception:
            pass
        preds = (s >= 0.5).astype(int)
        f1s.append(f1_score(yva, preds))

        # Coleta de cutoff Top-K deste fold (por vaga)
        df_va = pd.DataFrame({"y": yva, "s": s, "g": gva})
        for _, grp in df_va.groupby("g", sort=False):
            grp_sorted = grp.sort_values("s", ascending=False)
            if len(grp_sorted) == 0:
                continue
            idx = min(TARGET_K - 1, len(grp_sorted) - 1)
            kth_scores.append(float(grp_sorted["s"].iloc[idx]))

        t_fold = time.perf_counter() - t_fold0
        print(f"[Fold {fold}] NDCG@{TARGET_K}={nd:.4f} P@{TARGET_K}={p5:.4f} R@{TARGET_K}={r5:.4f} MRR={mrrv:.4f} F1@0.5={f1s[-1]:.4f} | tempo={_fmt_secs(t_fold)} (treino={len(tr)} valid={len(va)})")
        valid_folds += 1

    t_cv = time.perf_counter() - t_cv0
    print(f"[TIMER] Validação cruzada: { _fmt_secs(t_cv) } (folds válidos={valid_folds}/{n_splits})")

    if valid_folds == 0:
        print("[AVISO] Nenhum fold válido (classe única nos splits). Métricas serão 0.")
        metrics = {"NDCG@5_mean": 0.0, "F1_mean": 0.0, "ROC_AUC_mean": 0.0}
    else:
        metrics = {
            "NDCG@5_mean": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
            "F1_mean": float(sum(f1s) / len(f1s)) if f1s else 0.0,
            "ROC_AUC_mean": float(sum(rocs) / len(rocs)) if rocs else 0.0,
        }

    # Calibração do cutoff orientado a Top-K
    if kth_scores:
        threshold_topk = float(np.median(kth_scores))
    else:
        threshold_topk = 0.5  # fallback seguro
    print(f"[Cutoff] threshold_topk(median Kth score) = {threshold_topk:.4f}")

    print("[Metrics]", json.dumps(metrics, ensure_ascii=False, indent=2))

    # =======================
    # Fit final + salvamento
    # =======================
    t_fit0 = time.perf_counter()
    if len(np.unique(y)) < 2:
        print("[AVISO] Dataset completo com classe única — usando DummyClassifier(most_frequent).")
        pipe.set_params(clf=DummyClassifier(strategy="most_frequent"))
    pipe.fit(X, y)
    t_fit = time.perf_counter() - t_fit0
    print(f"[TIMER] Fit final (tudo): { _fmt_secs(t_fit) }")

    t_save0 = time.perf_counter()
    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(json.dumps({
        "features": ["text_concat via TF-IDF (word+char)"],
        "target": "y",
        "metrics": metrics,
        "ranking": {
            "target_k": TARGET_K,
            "threshold_topk": threshold_topk
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    t_save = time.perf_counter() - t_save0
    print(f"[TIMER] Persistência de artefatos: { _fmt_secs(t_save) }")

    t_total = time.perf_counter() - t0
    print(f"[TIMER] Tempo total do pipeline: { _fmt_secs(t_total) } (fim: { _now() })")

# === Salva baseline de features simples p/ drift ===
ref_features = pd.DataFrame({
    "cv_len": data["cv_pt"].fillna("").astype(str).str.len(),
    "job_len": (
        data["principais_atividades"].fillna("").astype(str) + " " +
        data["competencias"].fillna("").astype(str) + " " +
        data["observacoes"].fillna("").astype(str) + " " +
        data["titulo_vaga"].fillna("").astype(str)
    ).str.len(),
    # score não existe no treino; usar placeholder NaN
    "score": np.nan
})
ref_path = MODELS_DIR / "baseline_features.csv"
ref_features.to_csv(ref_path, index=False)
print(f"[Baseline] features salvas em: {ref_path}")


if __name__ == "__main__":
    main()