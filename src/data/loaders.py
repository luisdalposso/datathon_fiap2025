from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import pandas as pd

def _safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_applicants(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data.values() if isinstance(data, dict) else data:
        codigo = _safe_get(item, ["infos_basicas", "codigo_profissional"]) or _safe_get(item, ["informacoes_pessoais", "codigo_profissional"])
        nome = _safe_get(item, ["informacoes_pessoais", "nome"])
        area = _safe_get(item, ["informacoes_profissionais", "area_atuacao"])
        nivel_prof = _safe_get(item, ["informacoes_profissionais", "nivel_profissional"])
        nivel_acad = _safe_get(item, ["formacao_e_idiomas", "nivel_academico"])
        ing = _safe_get(item, ["formacao_e_idiomas", "nivel_ingles"])
        esp = _safe_get(item, ["formacao_e_idiomas", "nivel_espanhol"])
        cv_pt = _safe_get(item, ["curriculo", "cv_pt"]) or _safe_get(item, ["cv_pt"], "")
        rows.append({
            "applicant_id": str(codigo) if codigo is not None else None,
            "nome": nome,
            "area_atuacao": area,
            "nivel_profissional_cand": nivel_prof,
            "nivel_academico_cand": nivel_acad,
            "nivel_ingles_cand": ing,
            "nivel_espanhol_cand": esp,
            "cv_pt": cv_pt or "",
        })
    return pd.DataFrame(rows)

def load_jobs(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    if isinstance(data, dict):
        iterator = data.items()  # (job_id, item)
    else:
        # Caso raro: lista. Usa o índice como fallback de id.
        iterator = enumerate(data)

    for key, item in iterator:
        job_id = str(key)

        ib = item.get("informacoes_basicas", {}) if isinstance(item, dict) else {}
        perfil = item.get("perfil_vaga", {}) if isinstance(item, dict) else {}

        titulo = ib.get("titulo_vaga") or item.get("titulo") or ""
        cliente = ib.get("cliente") or item.get("cliente") or ""

        nivel_prof = perfil.get("nivel_profissional") or perfil.get("nivel profissional")
        nivel_acad = perfil.get("nivel_academico") or perfil.get("nivel academico")
        nivel_ing = perfil.get("nivel_ingles") or perfil.get("nivel ingles")
        nivel_esp = perfil.get("nivel_espanhol") or perfil.get("nivel espanhol")
        area = perfil.get("area_atuacao") or perfil.get("area atuacao")

        principais = item.get("principais_atividades") or ""
        comp = item.get("competencia_tecnicas_e_comportamentais") or item.get("competencias") or ""
        obs = item.get("demais_observacoes") or ""

        rows.append({
            "job_id": job_id,
            "titulo_vaga": titulo,
            "cliente": cliente,
            "nivel_profissional_vaga": nivel_prof,
            "nivel_academico_vaga": nivel_acad,
            "nivel_ingles_vaga": nivel_ing,
            "nivel_espanhol_vaga": nivel_esp,
            "area_atuacao_vaga": area,
            "principais_atividades": principais or "",
            "competencias": comp or "",
            "observacoes": obs or "",
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    # higienização extra: sem ids vazios e sem duplicatas
    df = df[df["job_id"].notna() & (df["job_id"] != "")]
    df = df.drop_duplicates("job_id", keep="first").reset_index(drop=True)
    return df


def load_prospects(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for job_id, job_obj in (data.items() if isinstance(data, dict) else enumerate(data)):
        title = job_obj.get("titulo") or job_obj.get("titulo_vaga")
        for p in job_obj.get("prospects", []):
            # aceita tanto a chave correta quanto a com typo
            situacao = p.get("situacao_candidato")
            if situacao is None:
                situacao = p.get("situacao_candidado")  # <- typo comum na base

            rows.append({
                "job_id": str(job_id),
                "titulo_vaga": title,
                "applicant_id": str(p.get("codigo")) if p.get("codigo") is not None else None,
                "situacao": situacao,
                "data_candidatura": p.get("data_candidatura"),
                "ultima_atualizacao": p.get("ultima_atualizacao"),
            })
    return pd.DataFrame(rows)