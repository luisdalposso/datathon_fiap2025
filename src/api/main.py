
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    ScoreRequest, ScoreResponse,
    RankCandidatesRequest, RankResponse, RankItem
)
import joblib, json, os
import numpy as np
from pathlib import Path
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

app = FastAPI(title="Decision Match API", version="0.3.1")

# CORS liberado para dev/local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "artifacts" / "model.joblib"
META_PATH = ROOT / "models" / "artifacts" / "metadata.json"

_model = None
_threshold_topk = 0.5
_target_k = 5

def _score_df(df: pd.DataFrame) -> np.ndarray:
    """Retorna scores [0,1] para um DataFrame no formato da pipeline."""
    if _model is None:
        return np.zeros(len(df), dtype=float)
    df = df.copy().fillna("")
    last = _model[-1]
    if hasattr(last, "predict_proba"):
        s = _model.predict_proba(df)[:, 1]
    elif hasattr(last, "decision_function"):
        dfu = _model.decision_function(df)
        s = (dfu - dfu.min()) / (dfu.max() - dfu.min() + 1e-9)
    else:
        s = _model.predict(df).astype(float)
    return np.clip(s, 0.0, 1.0)

@app.on_event("startup")
def load_model():
    global _model, _threshold_topk, _target_k
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            rk = meta.get("ranking", {})
            _threshold_topk = float(rk.get("threshold_topk", _threshold_topk))
            _target_k = int(rk.get("target_k", _target_k))
        except Exception:
            pass
    # overrides por env
    env_thr = os.getenv("THRESHOLD_TOPK")
    if env_thr:
        try:
            _threshold_topk = float(env_thr)
        except Exception:
            pass
    env_k = os.getenv("TARGET_K")
    if env_k and env_k.isdigit():
        _target_k = int(env_k)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "threshold_topk": _threshold_topk,
        "target_k": _target_k
    }

@app.get("/config")
def config():
    return {"threshold_topk": _threshold_topk, "target_k": _target_k}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest):
    Xdf = pd.DataFrame({
        "cv_pt": [payload.cv_pt or ""],
        "principais_atividades": [payload.principais_atividades or ""],
        "competencias": [payload.competencias or ""],
        "observacoes": [payload.observacoes or ""],
        "titulo_vaga": [payload.titulo_vaga or ""],
    }).fillna("")
    score_val = float(_score_df(Xdf)[0])
    return ScoreResponse(
        score=score_val,
        pass_by_threshold=score_val >= _threshold_topk,
        threshold_used=_threshold_topk,
    )

@app.post("/score-batch")
def score_batch(payload: list[ScoreRequest]):
    # retorna lista simples de {score, pass_by_threshold, threshold_used}
    rows = []
    for p in payload:
        rows.append({
            "cv_pt": p.cv_pt or "",
            "principais_atividades": p.principais_atividades or "",
            "competencias": p.competencias or "",
            "observacoes": p.observacoes or "",
            "titulo_vaga": p.titulo_vaga or "",
        })
    Xdf = pd.DataFrame(rows).fillna("")
    scores = _score_df(Xdf)
    thr = _threshold_topk
    out = []
    for s in scores:
        s = float(s)
        out.append({
            "score": s,
            "pass_by_threshold": s >= thr,
            "threshold_used": thr,
        })
    return out

@app.post("/rank-candidates", response_model=RankResponse)
def rank_candidates(payload: RankCandidatesRequest):
    # monta uma linha por candidato, repetindo o contexto da vaga
    rows = []
    ids, names = [], []
    for c in payload.candidates:
        ids.append(c.id)
        names.append(c.name)
        rows.append({
            "cv_pt": c.cv_pt or "",
            "principais_atividades": payload.principais_atividades or "",
            # combina competências/observações do job com as do candidato
            "competencias": " ".join(filter(None, [payload.competencias or "", c.competencias or ""])),
            "observacoes": " ".join(filter(None, [payload.observacoes or "", c.observacoes or ""])),
            "titulo_vaga": payload.titulo_vaga or "",
        })

    Xdf = pd.DataFrame(rows).fillna("")
    scores = _score_df(Xdf)

    # ordena por score desc
    order = np.argsort(-scores)
    k_target = payload.k if (payload.k and payload.k > 0) else _target_k

    # aplica threshold (se pedido), depois corta em K
    items_all = []
    thr = _threshold_topk
    use_thr = bool(payload.use_threshold)
    for idx in order:
        s = float(scores[idx])
        pass_thr = s >= thr if use_thr else True
        items_all.append(RankItem(
            id=str(ids[idx]) if ids[idx] is not None else None,
            name=str(names[idx]) if names[idx] is not None else None,
            score=s,
            pass_by_threshold=pass_thr
        ))

    if use_thr:
        items_all = [it for it in items_all if it.pass_by_threshold]

    items = items_all[:k_target]
    return RankResponse(
        items=items,
        used_k=len(items),
        threshold_used=thr
    )
