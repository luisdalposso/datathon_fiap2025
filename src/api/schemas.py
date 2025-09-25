
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import math

def _coerce_str(v):
    if v is None: return ""
    if isinstance(v, float) and math.isnan(v): return ""
    return v  # pydantic ainda pode converter para str se necessário

class ScoreRequest(BaseModel):
    cv_pt: Optional[str] = ""
    principais_atividades: Optional[str] = ""
    competencias: Optional[str] = ""
    observacoes: Optional[str] = ""
    titulo_vaga: Optional[str] = ""

    # limpa NaN/None em todos
    @field_validator("cv_pt","principais_atividades","competencias","observacoes","titulo_vaga", mode="before")
    @classmethod
    def _clean(cls, v): return _coerce_str(v)

class ScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    pass_by_threshold: bool
    threshold_used: float

class CandidatePayload(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    cv_pt: Optional[str] = ""
    competencias: Optional[str] = ""
    observacoes: Optional[str] = ""

    @field_validator("id","name","cv_pt","competencias","observacoes", mode="before")
    @classmethod
    def _clean(cls, v): return _coerce_str(v)

class RankCandidatesRequest(BaseModel):
    titulo_vaga: Optional[str] = ""
    principais_atividades: Optional[str] = ""
    competencias: Optional[str] = ""
    observacoes: Optional[str] = ""
    candidates: List[CandidatePayload]
    k: Optional[int] = None
    use_threshold: bool = True

    @field_validator("titulo_vaga","principais_atividades","competencias","observacoes", mode="before")
    @classmethod
    def _clean(cls, v): return _coerce_str(v)

class RankItem(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    score: float
    pass_by_threshold: bool

class RankResponse(BaseModel):
    items: List[RankItem]
    used_k: int
    threshold_used: float
