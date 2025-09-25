from fastapi import FastAPI
from .schemas import ScoreRequest, ScoreResponse
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Decision Match API", version="0.1.0")

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "artifacts" / "model.joblib"
_model = None

@app.on_event("startup")
def load_model():
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    else:
        _model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest):
    if _model is None:
        return ScoreResponse(score=0.0)
    # Monta um "DataFrame-like" mínimo para o pipeline
    X = {
        "cv_pt": [payload.cv_pt or ""],
        "principais_atividades": [payload.principais_atividades or ""],
        "competencias": [payload.competencias or ""],
        "observacoes": [payload.observacoes or ""],
        "titulo_vaga": [payload.titulo_vaga or ""],
    }
    # A pipeline aceita DataFrame; para manter leve, usamos dict; sklearn trata via __array__ em etapas customizadas
    # Porém nossa pipeline usa FunctionTransformer sobre DataFrame, então convertamos para pandas apenas aqui
    import pandas as pd
    Xdf = pd.DataFrame(X)
    if hasattr(_model[-1], "predict_proba"):
        s = _model.predict_proba(Xdf)[:, 1]
    else:
        if hasattr(_model[-1], "decision_function"):
            dfu = _model.decision_function(Xdf)
            s = (dfu - dfu.min()) / (dfu.max() - dfu.min() + 1e-9)
        else:
            s = _model.predict(Xdf)
    score = float(np.clip(s[0], 0.0, 1.0))
    return ScoreResponse(score=score)
