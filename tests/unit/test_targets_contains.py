import os, importlib
from importlib import reload

def test_positive_contains_mode(monkeypatch):
    # ativa modo "contém"
    monkeypatch.setenv("TREAT_POSITIVE_CONTAINS", "true")
    import src.labeling.targets as targets
    reload(targets)  # reavalia constantes à luz do env

    # Variedade com sufixo -> deve ser 1
    assert targets.map_status_to_label("Contratado pelo cliente - 22/09") == 1

    # volta para o padrão (evitar efeitos colaterais em outros testes)
    monkeypatch.delenv("TREAT_POSITIVE_CONTAINS", raising=False)
    reload(targets)
