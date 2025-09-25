from src.labeling.targets import map_status_to_label

def test_map_status_to_label():
    assert map_status_to_label("Contratado pelo Cliente") == 1
    assert map_status_to_label("Reprovado") == 0
    assert map_status_to_label("Encaminhado ao Requisitante") == 0
    assert map_status_to_label(None) == 0
