import json
from importlib import reload

import src.api.main as m


def test_load_model_with_bad_metadata_and_bad_env(tmp_path, monkeypatch):
    # metadata.json inválido (força o except)
    bad_meta = tmp_path / "metadata.json"
    bad_meta.write_text("{invalid-json", encoding="utf-8")

    monkeypatch.setattr(m, "META_PATH", bad_meta, raising=True)
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib", raising=True)

    # valores antes (para conferir que não foram corrompidos)
    old_thr = m._threshold_topk
    old_k = m._target_k

    # overrides inválidos (devem ser ignorados)
    monkeypatch.setenv("THRESHOLD_TOPK", "abc")
    monkeypatch.setenv("TARGET_K", "x9")

    m.load_model()
    assert m._threshold_topk == old_thr
    assert m._target_k == old_k
