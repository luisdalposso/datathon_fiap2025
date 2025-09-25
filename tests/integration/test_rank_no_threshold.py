# tests/integration/test_rank_no_threshold.py
from fastapi.testclient import TestClient
import src.api.main as m

client = TestClient(m.app)

def test_rank_candidates_without_threshold(monkeypatch):
    payload = {
        "titulo_vaga": "Backend Python",
        "principais_atividades": "APIs REST",
        "competencias": "Python; FastAPI",
        "observacoes": "",
        "k": 2,
        "use_threshold": False,   # <- cobre ramo sem threshold
        "candidates": [
            {"id": "1", "name": "Ana", "cv_pt": "Python FastAPI", "competencias": "", "observacoes": ""},
            {"id": "2", "name": "Bia", "cv_pt": "SQL Docker", "competencias": "", "observacoes": ""},
            {"id": "3", "name": "Caio", "cv_pt": "NoSQL", "competencias": "", "observacoes": ""},
        ],
    }
    r = client.post("/rank-candidates", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "items" in data and data["used_k"] <= 2
