from typing import Optional, List
from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    cv_pt: Optional[str] = ""
    principais_atividades: Optional[str] = ""
    competencias: Optional[str] = ""
    observacoes: Optional[str] = ""
    titulo_vaga: Optional[str] = ""

class ScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    pass_by_threshold: bool
    threshold_used: float

class CandidatePayload(BaseModel):
    id: Optional[str] = None
    cv_pt: Optional[str] = ""
    competencias: Optional[str] = ""
    observacoes: Optional[str] = ""

class RankCandidatesRequest(BaseModel):
    # contexto da vaga
    titulo_vaga: Optional[str] = ""
    principais_atividades: Optional[str] = ""
    competencias: Optional[str] = ""
    observacoes: Optional[str] = ""
    # candidatos a ranquear
    candidates: List[CandidatePayload]
    # parâmetros
    k: Optional[int] = None
    use_threshold: bool = True

class RankItem(BaseModel):
    id: Optional[str] = None
    score: float
    pass_by_threshold: bool

class RankResponse(BaseModel):
    items: List[RankItem]
    used_k: int
    threshold_used: float