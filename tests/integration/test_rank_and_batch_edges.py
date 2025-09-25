from fastapi.testclient import TestClient
import src.api.main as m

client = TestClient(m.app)

def test_rank_candidates_empty_list():
    payload = {
        "titulo_vaga": "Backend",
        "principais_atividades": "",
        "competencias": "",
        "observacoes": "",
        "k": 5,
        "use_threshold": True,
        "candidates": [],
    }
    r = client.post("/rank-candidates", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["used_k"] == 0
    assert data["items"] == []

def test_score_batch_empty():
    r = client.post("/score-batch", json=[])
    assert r.status_code == 200
    assert r.json() == []
