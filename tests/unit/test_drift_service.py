import os
import csv
from importlib import reload
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    # MONITORING_DIR e DRIFT_REPORTS_DIR para o serviço
    mon_dir = tmp_path / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = mon_dir / "drift_reports"
    monkeypatch.setenv("MONITORING_DIR", str(mon_dir))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(reports_dir))
    return {"mon_dir": mon_dir, "reports_dir": reports_dir, "tmp_path": tmp_path}


def _seed_baseline(baseline_path: Path):
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "cv_len": [10, 20, 30],
            "job_len": [100, 120, 140],
            "score": [0.1, 0.5, 0.9],
        }
    )
    df.to_csv(baseline_path, index=False)


def _seed_requests_log(log_file: Path, n: int = 250):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "endpoint", "cv_len", "job_len", "score"])
        for i in range(n):
            w.writerow([i, "/score", 80 + (i % 10), 300 + (i % 25), (i % 10) / 10.0])


def test_drift_service_health_and_metrics(tmp_env, monkeypatch):
    # Import tardio do módulo com env já setado
    import src.monitoring.drift_service as drift_service
    reload(drift_service)  # garante que pegue os env atuais

    # Redireciona ARTIFACTS/BASELINE para um caminho de teste
    artifacts = tmp_env["tmp_path"] / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    baseline = artifacts / "baseline_features.csv"

    drift_service.ARTIFACTS = artifacts
    drift_service.BASELINE = baseline

    # Cria baseline e requests_log para o /health conseguir ver arquivos
    _seed_baseline(baseline)
    log_file = tmp_env["mon_dir"] / "requests_log.csv"
    _seed_requests_log(log_file, n=220)

    # Garante que o módulo enxergue o novo LOG_FILE
    drift_service.MON_DIR = tmp_env["mon_dir"]
    drift_service.LOG_FILE = log_file

    client = TestClient(drift_service.app)

    # /health deve responder 200 e indicar existência dos arquivos
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("baseline_exists") is True
    assert body.get("log_exists") is True
    assert "reports_dir" in body

    # /metrics deve responder 200 (mesmo sem evidently instalado)
    m = client.get("/metrics")
    assert m.status_code == 200
    text = m.text
    # Exposição padrão do prometheus_client (python_info) já valida o endpoint
    assert "python_info" in text

    # Executa a rotina de cálculo (não falha se evidently não estiver instalado)
    drift_service.compute_and_export()
