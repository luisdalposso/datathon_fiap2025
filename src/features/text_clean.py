import regex as re
from unidecode import unidecode

_sp = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unidecode(str(s).lower())
    s = re.sub(r"[^a-z0-9\s\+\#\.\-_/]", " ", s)  # mantem +, #, ., -, _, /
    s = _sp.sub(" ", s).strip()
    return s
