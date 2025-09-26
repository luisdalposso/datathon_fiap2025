from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_config_endpoint_shape():
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    # chaves obrigatórias atuais
    assert "threshold_topk" in body
    assert "target_k" in body
    # 'model_loaded' pode ou não existir (dependendo da implementação atual)
    if "model_loaded" in body:
        assert isinstance(body["model_loaded"], bool)


def test_rank_candidates_contract():
    payload = {
        "titulo_vaga": "Backend Python",
        "principais_atividades": "Construir e otimizar APIs",
        "competencias": "Python; FastAPI; SQL",
        "observacoes": "",
        "candidates": [
            {"id": "1", "name": "A", "cv_pt": "APIs e Docker", "competencias": "Python", "observacoes": ""},
            {"id": "2", "name": "B", "cv_pt": "Dados e ETL", "competencias": "Python; SQL", "observacoes": ""},
            {"id": "3", "name": "C", "cv_pt": "Microsserviços", "competencias": "Python; FastAPI", "observacoes": ""}
        ],
        "k": 2,
        "use_threshold": True
    }
    r = client.post("/rank-candidates", json=payload)
    assert r.status_code == 200
    body = r.json()
    # contrato atual: objeto com 'items' + metadados
    assert isinstance(body, dict)
    assert "items" in body and isinstance(body["items"], list)
    assert "threshold_used" in body
    assert "used_k" in body


def test_metrics_endpoint_exposes_counters():
    # cobertura adicional e verificação útil do contrato de observabilidade
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "dm_api_requests_total" in text
