# src/api/main.py
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

import csv
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from .metrics import REQUESTS, LATENCY
from .schemas import (
    ScoreRequest,
    ScoreResponse,
    RankCandidatesRequest,
    RankResponse,
    RankItem,
)

# =========================
# Paths, Globals & Config
# =========================
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "artifacts" / "model.joblib"
META_PATH = ROOT / "models" / "artifacts" / "metadata.json"

# Diretório para logs de monitoramento (montado via Docker)
MONITORING_DIR = os.getenv("MONITORING_DIR", "/monitoring")
LOG_FILE = os.path.join(MONITORING_DIR, "requests_log.csv")

_model: Optional[object] = None
_threshold_topk: float = 0.5
_target_k: int = 5


# =========================
# Helpers internos
# =========================
def _score_df(df: pd.DataFrame) -> np.ndarray:
    """Retorna scores [0,1] para um DataFrame no formato da pipeline."""
    global _model
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


def load_model():
    """Carrega modelo/metadata e aplica overrides de ambiente."""
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
            # metadata inválida: mantém defaults/overrides
            pass

    # overrides por env
    env_thr = os.getenv("THRESHOLD_TOPK")
    if env_thr:
        try:
            _threshold_topk = float(env_thr)
        except Exception:
            pass

    env_k = os.getenv("TARGET_K")
    if env_k:
        try:
            _target_k = int(env_k)
        except Exception:
            pass


def _init_monitoring():
    """Garante diretório/arquivo de log para drift."""
    if not MONITORING_DIR:
        return
    try:
        os.makedirs(MONITORING_DIR, exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts", "endpoint", "cv_len", "job_len", "score"])
    except Exception:
        # não bloqueia o app se não conseguir criar diretório/arquivo
        pass


def _append_monitor_rows(rows):
    """Anexa linhas no CSV de monitoramento; falhas são silenciosas."""
    if not MONITORING_DIR:
        return
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)
    except Exception:
        pass


# =========================
# App factory (lifespan)
# =========================
def _build_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Inicialização
        load_model()
        _init_monitoring()
        yield
        # Finalização: nada por enquanto

    app = FastAPI(title="Decision Match API", version="0.3.3", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = _build_app()


# =========================
# Middleware de métricas
# =========================
@app.middleware("http")
async def metrics_and_access_log(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    status_code = 500
    try:
        response: StarletteResponse = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        dur = time.perf_counter() - start
        try:
            LATENCY.labels(endpoint=path).observe(dur)
            REQUESTS.labels(endpoint=path, method=method, status=str(status_code)).inc()
        except Exception:
            # nunca quebre a requisição por falha de métrica
            pass


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "threshold_topk": _threshold_topk,
        "target_k": _target_k,
    }


@app.get("/config")
def config():
    return {"threshold_topk": _threshold_topk, "target_k": _target_k}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest):
    Xdf = pd.DataFrame(
        {
            "cv_pt": [payload.cv_pt or ""],
            "principais_atividades": [payload.principais_atividades or ""],
            "competencias": [payload.competencias or ""],
            "observacoes": [payload.observacoes or ""],
            "titulo_vaga": [payload.titulo_vaga or ""],
        }
    ).fillna("")
    score_val = float(_score_df(Xdf)[0])

    # ===== Log leve para drift (sem PII) =====
    try:
        cv_len = len(payload.cv_pt or "")
        job_len = len(
            " ".join(
                [
                    payload.principais_atividades or "",
                    payload.competencias or "",
                    payload.observacoes or "",
                    payload.titulo_vaga or "",
                ]
            )
        )
        _append_monitor_rows([[time.time(), "/score", cv_len, job_len, score_val]])
    except Exception:
        pass

    return ScoreResponse(
        score=score_val,
        pass_by_threshold=score_val >= _threshold_topk,
        threshold_used=_threshold_topk,
    )


@app.post("/score-batch")
def score_batch(payload: List[ScoreRequest]):
    """Retorna lista simples de {score, pass_by_threshold, threshold_used}."""
    rows = []
    for p in payload:
        rows.append(
            {
                "cv_pt": p.cv_pt or "",
                "principais_atividades": p.principais_atividades or "",
                "competencias": p.competencias or "",
                "observacoes": p.observacoes or "",
                "titulo_vaga": p.titulo_vaga or "",
            }
        )
    if not rows:
        return []
    Xdf = pd.DataFrame(rows).fillna("")
    scores = _score_df(Xdf)
    thr = _threshold_topk

    out = []
    for s in scores:
        s = float(s)
        out.append(
            {
                "score": s,
                "pass_by_threshold": s >= thr,
                "threshold_used": thr,
            }
        )
    return out


@app.post("/rank-candidates", response_model=RankResponse)
def rank_candidates(payload: RankCandidatesRequest):
    """Ranqueia candidatos para uma vaga, aplicando threshold opcional e top-K."""
    # monta uma linha por candidato, repetindo o contexto da vaga
    rows = []
    ids, names = [], []
    for c in payload.candidates:
        ids.append(c.id)
        names.append(c.name)
        rows.append(
            {
                "cv_pt": c.cv_pt or "",
                "principais_atividades": payload.principais_atividades or "",
                # combina competências/observações do job com as do candidato
                "competencias": " ".join(
                    filter(None, [payload.competencias or "", c.competencias or ""])
                ),
                "observacoes": " ".join(
                    filter(None, [payload.observacoes or "", c.observacoes or ""])
                ),
                "titulo_vaga": payload.titulo_vaga or "",
            }
        )

    Xdf = pd.DataFrame(rows).fillna("")
    scores = _score_df(Xdf)

    # ===== Log leve para drift (sem PII) — 1 linha por candidato =====
    try:
        job_txt = (
            Xdf["principais_atividades"].astype(str)
            + " "
            + Xdf["competencias"].astype(str)
            + " "
            + Xdf["observacoes"].astype(str)
            + " "
            + Xdf["titulo_vaga"].astype(str)
        )
        ts = time.time()
        to_log = []
        for i in range(len(Xdf)):
            cv_len_i = len(str(Xdf.iloc[i]["cv_pt"]))
            job_len_i = len(str(job_txt.iloc[i]))
            to_log.append([ts, "/rank-candidates", cv_len_i, job_len_i, float(scores[i])])
        if to_log:
            _append_monitor_rows(to_log)
    except Exception:
        pass

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
        items_all.append(
            RankItem(
                id=str(ids[idx]) if ids[idx] is not None else None,
                name=str(names[idx]) if names[idx] is not None else None,
                score=s,
                pass_by_threshold=pass_thr,
            )
        )

    if use_thr:
        items_all = [it for it in items_all if it.pass_by_threshold]

    items = items_all[:k_target]
    return RankResponse(items=items, used_k=len(items), threshold_used=thr)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
