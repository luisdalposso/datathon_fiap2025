# tests/integration/test_env_and_metadata.py
import json
from pathlib import Path
from importlib import reload

import numpy as np
import pandas as pd
import pytest

import src.api.main as m


def test_load_model_reads_metadata_and_env(tmp_path, monkeypatch):
    # cria metadata.json com valores customizados
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps({
        "ranking": {"threshold_topk": 0.7, "target_k": 3}
    }, ensure_ascii=False), encoding="utf-8")

    # aponta o módulo para esse caminho e evita carregar modelo real
    monkeypatch.setattr(m, "META_PATH", meta_path, raising=True)
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib", raising=True)

    # limpa overrides e carrega
    monkeypatch.delenv("THRESHOLD_TOPK", raising=False)
    monkeypatch.delenv("TARGET_K", raising=False)
    m.load_model()
    assert m._threshold_topk == 0.7
    assert m._target_k == 3

    # agora testa overrides de ambiente
    monkeypatch.setenv("THRESHOLD_TOPK", "0.33")
    monkeypatch.setenv("TARGET_K", "7")
    m.load_model()
    assert m._threshold_topk == 0.33
    assert m._target_k == 7
