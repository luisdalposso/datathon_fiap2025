from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Applicant:
    applicant_id: str
    nome: Optional[str]
    area_atuacao: Optional[str]
    nivel_profissional: Optional[str]
    nivel_academico: Optional[str]
    nivel_ingles: Optional[str]
    nivel_espanhol: Optional[str]
    cv_pt: Optional[str]

@dataclass
class Job:
    job_id: str
    titulo: Optional[str]
    cliente: Optional[str]
    nivel_profissional: Optional[str]
    nivel_academico: Optional[str]
    nivel_ingles: Optional[str]
    nivel_espanhol: Optional[str]
    area_atuacao: Optional[str]
    principais_atividades: Optional[str]
    competencias: Optional[str]
    observacoes: Optional[str]

# Generic dict payloads for API can map to these dataclasses later if desired.
