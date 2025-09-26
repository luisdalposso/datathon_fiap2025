import sys
import types
from pathlib import Path
from importlib import reload
import csv
import pandas as pd

def test_compute_handles_html_and_asdict_exceptions(tmp_path):
    # Fake Evidently que explode em save_html e as_dict
    fake_report_mod = types.ModuleType("evidently.report")
    fake_preset_mod = types.ModuleType("evidently.metric_preset")

    class FakeReport:
        def __init__(self, metrics):
            pass
        def run(self, reference_data, current_data):
            return None
        def save_html(self, out):
            raise RuntimeError("boom on save_html")
        def as_dict(self):
            raise RuntimeError("boom on as_dict")

    class FakePreset:
        pass

    sys.modules["evidently"] = types.ModuleType("evidently")
    sys.modules["evidently.report"] = fake_report_mod
    sys.modules["evidently.metric_preset"] = fake_preset_mod
    fake_report_mod.Report = FakeReport
    fake_preset_mod.DataDriftPreset = FakePreset

    import src.monitoring.drift_service as drift_service
    reload(drift_service)

    # preparar baseline/log suficientes
    artifacts = tmp_path / "artifacts"
    baseline = artifacts / "baseline_features.csv"
    mon = tmp_path / "monitoring"
    reports = mon / "drift_reports"
    logf = mon / "requests_log.csv"

    artifacts.mkdir(parents=True, exist_ok=True)
    mon.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"cv_len":[10]*210, "job_len":[100]*210, "score":[0.5]*210}).to_csv(baseline, index=False)

    with logf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","endpoint","cv_len","job_len","score"])
        for i in range(210):
            w.writerow([i, "/score", 12+i%3, 200+i%5, (i%10)/10])

    drift_service.ARTIFACTS = artifacts
    drift_service.BASELINE = baseline
    drift_service.MON_DIR = mon
    drift_service.REPORTS_DIR = reports
    drift_service.LOG_FILE = logf

    # não deve lançar erro, mesmo com as exceções internas
    drift_service.compute_and_export()

    # como save_html falhou, não deve haver arquivo
    assert list(reports.glob("drift_*.html")) == []
