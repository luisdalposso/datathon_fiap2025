# src/labeling/targets.py
from typing import Optional
from unidecode import unidecode
import os
import re

# -------------------------------------------------------------------
# Configuração
# -------------------------------------------------------------------
# Por padrão, só consideramos POSITIVO quando a situação for IGUAL,
# após normalização (minúsculas, sem acentos, trim).
# Se você quiser aceitar "contém" (ex.: "encaminhado ao requisitante - 22/09"),
# defina a variável de ambiente: TREAT_POSITIVE_CONTAINS=true
TREAT_POSITIVE_CONTAINS = os.getenv("TREAT_POSITIVE_CONTAINS", "false").lower() == "true"

# Situações consideradas positivas (após normalização)
POSITIVE_EXACT = {
    "entrevista com cliente",
    "encaminhado ao requisitante",
    "contratado pela decision",
}

# Se usar modo "contém", estas mesmas frases serão buscadas como substrings.
# (mantenha coerente com POSITIVE_EXACT)
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
    Mapeia a situação para rótulo binário:
      1 -> positivo
      0 -> negativo/demais casos
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