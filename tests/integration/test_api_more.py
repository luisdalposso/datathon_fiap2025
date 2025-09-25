from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_config_and_metrics():
    r = client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "threshold_topk" in data and "target_k" in data

    m = client.get("/metrics")
    assert m.status_code == 200
    assert "text/plain" in m.headers.get("content-type", "")

def test_score_batch_and_rank():
    batch = [
        {
            "cv_pt": "Dev Python com FastAPI e Docker",
            "titulo_vaga": "Backend Python",
            "principais_atividades": "Criar APIs REST",
            "competencias": "Python; FastAPI; Docker; SQL",
        },
        {
            "cv_pt": "Analista de dados com Pandas e sklearn",
            "titulo_vaga": "Cientista de Dados",
            "principais_atividades": "Modelagem",
            "competencias": "Pandas; scikit-learn; ML",
        },
    ]
    rb = client.post("/score-batch", json=batch)
    assert rb.status_code == 200
    assert isinstance(rb.json(), list)
    assert "score" in rb.json()[0]

    rank_payload = {
        "titulo_vaga": "Backend Python",
        "principais_atividades": "APIs REST",
        "competencias": "Python; FastAPI; Docker",
        "observacoes": "",
        "k": 3,
        "use_threshold": True,
        "candidates": [
            {"id": "1", "name": "Alice", "cv_pt": "Pythonista FastAPI", "competencias": "Python", "observacoes": ""},
            {"id": "2", "name": "Bob", "cv_pt": "SQL Docker", "competencias": "SQL; Docker", "observacoes": ""},
            {"id": "3", "name": "Carol", "cv_pt": "NoSQL", "competencias": "MongoDB", "observacoes": ""},
        ],
    }
    rr = client.post("/rank-candidates", json=rank_payload)
    assert rr.status_code == 200
    out = rr.json()
    assert "items" in out and "used_k" in out and "threshold_used" in out
