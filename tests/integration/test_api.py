from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_score():
    r = client.post("/score", json={
        "cv_pt": "Desenvolvedor Python com experiência em FastAPI e Docker.",
        "titulo_vaga": "Desenvolvedor Backend Python",
        "principais_atividades": "Construir e manter APIs REST.",
        "competencias": "Python; FastAPI; Docker; SQL",
    })
    assert r.status_code == 200
    assert "score" in r.json()
