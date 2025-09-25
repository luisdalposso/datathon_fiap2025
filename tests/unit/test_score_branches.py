# tests/unit/test_score_branches.py
import numpy as np
import pandas as pd
from importlib import reload

import src.api.main as m


class FakePipelineProba:
    def __getitem__(self, idx):  # permite m._model[-1]
        return self
    def predict_proba(self, X):
        # coluna 0 = 1 - p, coluna 1 = p
        n = len(X)
        p = np.linspace(0.1, 0.9, n)
        return np.c_[1 - p, p]

class FakePipelineDecision:
    def __getitem__(self, idx):
        return self
    def decision_function(self, X):
        # valores brutos que serão normalizados para [0,1]
        n = len(X)
        return np.linspace(-2.0, 2.0, n)

class FakePipelinePredict:
    def __getitem__(self, idx):
        return self
    def predict(self, X):
        # já retorna algo que vira float e é clipado para [0,1]
        n = len(X)
        return np.linspace(0.0, 1.0, n)


def _mk_df(n=5):
    return pd.DataFrame({
        "cv_pt": ["a"] * n,
        "principais_atividades": ["b"] * n,
        "competencias": ["c"] * n,
        "observacoes": ["d"] * n,
        "titulo_vaga": ["e"] * n,
    })

def test_score_branch_predict_proba(monkeypatch):
    monkeypatch.setattr(m, "_model", FakePipelineProba(), raising=True)
    s = m._score_df(_mk_df(5))
    assert (s >= 0).all() and (s <= 1).all()
    assert np.isclose(s[0], 0.1, atol=1e-6)

def test_score_branch_decision_function(monkeypatch):
    monkeypatch.setattr(m, "_model", FakePipelineDecision(), raising=True)
    s = m._score_df(_mk_df(5))
    # após normalização, extremos devem virar 0 e 1
    assert np.isclose(s.min(), 0.0, atol=1e-6)
    assert np.isclose(s.max(), 1.0, atol=1e-6)

def test_score_branch_predict(monkeypatch):
    monkeypatch.setattr(m, "_model", FakePipelinePredict(), raising=True)
    s = m._score_df(_mk_df(5))
    assert np.isclose(s[0], 0.0, atol=1e-6)
    assert np.isclose(s[-1], 1.0, atol=1e-6)

def test_score_when_model_is_none(monkeypatch):
    # cobre retorno de zeros
    monkeypatch.setattr(m, "_model", None, raising=True)
    s = m._score_df(_mk_df(3))
    assert np.allclose(s, 0.0)
