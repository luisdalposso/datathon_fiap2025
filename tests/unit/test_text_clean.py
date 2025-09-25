from src.features.text_clean import normalize_text

def test_normalize_text():
    assert normalize_text("Olá, Mundo! C++ #dev") == "ola mundo c++ #dev"
    assert normalize_text(None) == ""
