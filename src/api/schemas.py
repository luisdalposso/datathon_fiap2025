from pydantic import BaseModel, Field
from typing import Optional

class ScoreRequest(BaseModel):
    # Campos mínimos para escorar um par vaga×candidato
    cv_pt: Optional[str] = ""
    principais_atividades: Optional[str] = ""
    competencias: Optional[str] = ""
    observacoes: Optional[str] = ""
    titulo_vaga: Optional[str] = ""

class ScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
