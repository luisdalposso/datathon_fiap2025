from pathlib import Path
import csv
import pandas as pd
from importlib import reload

def test_load_baseline_adds_missing_columns(tmp_path):
    import src.monitoring.drift_service as drift_service
    reload(drift_service)

    # baseline com só uma coluna; o loader deve criar as demais como 0
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    baseline = artifacts / "baseline_features.csv"
    pd.DataFrame({"cv_len": [5, 10, 15]}).to_csv(baseline, index=False)

    drift_service.ARTIFACTS = artifacts
    drift_service.BASELINE = baseline

    df = drift_service._load_baseline()
    assert list(df.columns) == ["cv_len", "job_len", "score"]
    assert (df["job_len"] == 0).all()
    assert (df["score"] == 0).all()

def test_load_current_window_casts_and_drops(tmp_path):
    import src.monitoring.drift_service as drift_service
    reload(drift_service)

    mon = tmp_path / "monitoring"
    logf = mon / "requests_log.csv"
    mon.mkdir(parents=True, exist_ok=True)

    # mistura valores válidos e inválidos para exercitar to_numeric + dropna
    with logf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "endpoint", "cv_len", "job_len", "score"])
        # algumas linhas inválidas
        w.writerow([0, "/score", "x", 100, 0.1])
        w.writerow([1, "/score", 12, "y", 0.2])
        w.writerow([2, "/score", 15, 110, "z"])
        # linhas válidas (>=200 para passar min_rows)
        for i in range(205):
            w.writerow([i+3, "/score", 20 + (i % 3), 200 + (i % 5), (i % 10) / 10])

    drift_service.MON_DIR = mon
    drift_service.LOG_FILE = logf

    df = drift_service._load_current_window()
    # as 3 primeiras caem fora; o resto entra
    assert df is not None
    assert len(df) >= 200
    assert df["cv_len"].dtype.kind in "fi"
    assert df["job_len"].dtype.kind in "fi"
    assert df["score"].dtype.kind in "fi"
