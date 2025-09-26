# src/monitoring/drift_service.py
import os
import time
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# ---- Paths/Config ----
MON_DIR = Path(os.getenv("MONITORING_DIR", "/monitoring"))
ARTIFACTS = Path("/app/models/artifacts")
BASELINE = ARTIFACTS / "baseline_features.csv"
LOG_FILE = MON_DIR / "requests_log.csv"

# >>> gravar relatórios no diretório de monitoramento (RW), não em artifacts (RO)
REPORTS_DIR = Path(os.getenv("DRIFT_REPORTS_DIR", str(MON_DIR / "drift_reports")))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Prometheus Gauges ----
DRIFT_PVAL = Gauge("dm_drift_p_value", "Drift p-value por feature", ["feature"])
DRIFT_FLAG = Gauge("dm_drift_detected", "1 se drift detectado na feature", ["feature"])

# ---- App ----
app = FastAPI(title="Drift Monitor", version="0.1")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "baseline_exists": BASELINE.exists(),
        "log_exists": LOG_FILE.exists(),
        "reports_dir": str(REPORTS_DIR),
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---- IO helpers (leves) ----
def _load_baseline():
    if not BASELINE.exists():
        return None
    try:
        df = pd.read_csv(BASELINE)
        cols = ["cv_len", "job_len", "score"]
        for c in cols:
            if c not in df:
                df[c] = 0
        return df[cols]
    except Exception:
        return None

def _load_current_window(limit_rows: int = 5000, min_rows: int = 200):
    if not LOG_FILE.exists():
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        if len(df) > limit_rows:
            df = df.tail(limit_rows)
        cur = pd.DataFrame({
            "cv_len": pd.to_numeric(df["cv_len"], errors="coerce"),
            "job_len": pd.to_numeric(df["job_len"], errors="coerce"),
            "score":  pd.to_numeric(df["score"], errors="coerce"),
        }).dropna()
        if len(cur) < min_rows:
            return None
        return cur
    except Exception:
        return None

# ---- Core (lazy import Evidently) ----
def compute_and_export():
    ref = _load_baseline()
    cur = _load_current_window()
    if ref is None or cur is None:
        return

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        common = ["cv_len", "job_len", "score"]
        report.run(reference_data=ref[common].copy(), current_data=cur[common].copy())

        # salva HTML em /monitoring/drift_reports
        ts = int(time.time())
        out_html = REPORTS_DIR / f"drift_{ts}.html"
        try:
            report.save_html(str(out_html))
        except Exception:
            pass

        # exporta métricas
        try:
            as_dict = report.as_dict()
            for sec in as_dict.get("metrics", []):
                res = sec.get("result", {})
                per_col = res.get("drift_by_columns", {})
                for col, info in per_col.items():
                    p = info.get("p_value")
                    drifted = 1.0 if info.get("drift_detected") else 0.0
                    if p is not None:
                        DRIFT_PVAL.labels(feature=col).set(float(p))
                        DRIFT_FLAG.labels(feature=col).set(drifted)
        except Exception:
            pass

    except Exception:
        # qualquer falha do Evidently não deve derrubar o serviço
        pass

def _loop():
    while True:
        compute_and_export()
        time.sleep(60)

def start_bg_loop():
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

# inicia loop em background; /metrics é servido pelo uvicorn:8001
start_bg_loop()
