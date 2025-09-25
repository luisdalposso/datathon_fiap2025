# src/labeling/targets.py
from typing import Optional
from unidecode import unidecode
import os

# -------------------------------------------------------------------
# Configuração
# -------------------------------------------------------------------
TREAT_POSITIVE_CONTAINS = os.getenv("TREAT_POSITIVE_CONTAINS", "false").lower() == "true"

# Situações consideradas positivas (após normalização)
POSITIVE_EXACT = {
    "entrevista com cliente",
    "contratado pela decision",
    "contratado pelo cliente",
}

# Se usar modo "contém", estas mesmas frases serão buscadas como substrings.
POSITIVE_CONTAINS = tuple(POSITIVE_EXACT)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    """Normaliza string: sem acentos, minúsculas, espaços trimados."""
    return unidecode((s or "").strip().lower())

# -------------------------------------------------------------------
# API principal
# -------------------------------------------------------------------
def map_status_to_label(status: Optional[str]) -> int:
    """
    1 -> positivo; 0 -> demais casos.
    Regras:
      - Normaliza a string.
      - Se igual a uma das POSITIVE_EXACT -> 1
      - Se TREAT_POSITIVE_CONTAINS=true e contiver qualquer item -> 1
      - Caso contrário -> 0
    """
    s = _norm(status)
    if s in POSITIVE_EXACT:
        return 1
    if TREAT_POSITIVE_CONTAINS and any(p in s for p in POSITIVE_CONTAINS):
        return 1
    return 0