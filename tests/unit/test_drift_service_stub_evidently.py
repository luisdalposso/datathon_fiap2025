import sys
import types
import csv
from importlib import reload
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
import time


def _seed_baseline(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cv_len":[10,20,30], "job_len":[100,120,140], "score":[0.1,0.5,0.9]}).to_csv(path, index=False)


def _seed_requests_log(path: Path, n=220):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "endpoint", "cv_len", "job_len", "score"])
        for i in range(n):
            w.writerow([i, "/score", 50 + (i % 5), 200 + (i % 11), (i % 10) / 10.0])


def test_drift_compute_with_mocked_evidently(tmp_path, monkeypatch):
    # --- mock de evidently.* para não depender da lib real
    fake_report_mod = types.ModuleType("evidently.report")
    fake_preset_mod = types.ModuleType("evidently.metric_preset")

    class FakeReport:
        def __init__(self, metrics):
            self._metrics = metrics
        def run(self, reference_data, current_data):
            # simula algum processamento
            return None
        def save_html(self, out):
            Path(out).write_text("<html>ok</html>", encoding="utf-8")
        def as_dict(self):
            # estrutura mínima usada pelo código para exportar métricas
            return {
                "metrics":[
                    {
                        "result":{
                            "drift_by_columns": {
                                "cv_len": {"p_value": 0.03, "drift_detected": True},
                                "job_len": {"p_value": 0.2, "drift_detected": False},
                                "score":  {"p_value": 0.5, "drift_detected": False},
                            }
                        }
                    }
                ]
            }

    class FakePreset:
        pass

    fake_report_mod.Report = FakeReport
    fake_preset_mod.DataDriftPreset = FakePreset

    sys.modules["evidently"] = types.ModuleType("evidently")
    sys.modules["evidently.report"] = fake_report_mod
    sys.modules["evidently.metric_preset"] = fake_preset_mod

    # Importa o serviço com envs temporários
    import src.monitoring.drift_service as drift_service
    reload(drift_service)

    # Redireciona paths para a temp
    artifacts = tmp_path / "artifacts"
    baseline = artifacts / "baseline_features.csv"
    mon_dir = tmp_path / "monitoring"
    reports_dir = mon_dir / "drift_reports"
    log_file = mon_dir / "requests_log.csv"

    drift_service.ARTIFACTS = artifacts
    drift_service.BASELINE = baseline
    drift_service.MON_DIR = mon_dir
    drift_service.REPORTS_DIR = reports_dir
    drift_service.LOG_FILE = log_file

    # Semeia baseline e log (>=200 linhas para passar no min_rows)
    _seed_baseline(baseline)
    _seed_requests_log(log_file, n=230)

    # Executa o cálculo (vai usar FakeReport/FakePreset)
    drift_service.compute_and_export()

    # Verifica que salvou pelo menos um HTML
    htmls = sorted(reports_dir.glob("drift_*.html"))
    assert len(htmls) >= 1

    # Verifica que as métricas foram publicadas no /metrics do registry próprio
    client = TestClient(drift_service.app)
    m = client.get("/metrics")
    assert m.status_code == 200
    text = m.text
    assert 'dm_drift_p_value{feature="cv_len"} 0.03' in text
    assert 'dm_drift_detected{feature="cv_len"} 1' in text
    assert 'dm_drift_p_value{feature="job_len"} 0.2' in text
    assert 'dm_drift_detected{feature="job_len"} 0' in text


def test_drift_no_data_short_circuit(tmp_path, monkeypatch):
    # Recarrega serviço "limpo"
    import src.monitoring.drift_service as drift_service
    reload(drift_service)

    artifacts = tmp_path / "artifacts"
    baseline = artifacts / "baseline_features.csv"
    mon_dir = tmp_path / "monitoring"
    log_file = mon_dir / "requests_log.csv"

    drift_service.ARTIFACTS = artifacts
    drift_service.BASELINE = baseline
    drift_service.MON_DIR = mon_dir
    drift_service.LOG_FILE = log_file

    # Sem baseline e sem log -> compute_and_export retorna cedo sem lançar exceção
    drift_service.compute_and_export()

    # /health deve responder ok mesmo sem arquivos
    client = TestClient(drift_service.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["baseline_exists"] is False
    assert body["log_exists"] is False
