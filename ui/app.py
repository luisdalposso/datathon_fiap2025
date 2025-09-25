
# ui/app.py
import streamlit as st
import pandas as pd
import requests, json, csv, io

API = "http://127.0.0.1:8000/rank-candidates"

st.set_page_config(page_title="Decision Match — Ranking", layout="wide")
st.title("Decision Match — Ranking de Candidatos por Vaga")

# ---------------- Sidebar: Job context ----------------
with st.sidebar:
    st.header("Contexto da vaga")
    titulo = st.text_input("Título da vaga", "Backend Python")
    atividades = st.text_area("Principais atividades", "APIs REST; integrações")
    comp_job = st.text_input("Competências da vaga", "Python; FastAPI; Docker; SQL")
    obs_job = st.text_input("Observações", "Postgres desejável")
    k = st.number_input("Top-K", min_value=1, max_value=50, value=5, step=1)
    use_threshold = st.checkbox("Aplicar threshold calibrado (metadata.json)", value=True)

# ---------------- Helpers ----------------
def _s(v):
    import math
    if v is None: return ""
    if isinstance(v, float) and math.isnan(v): return ""
    return str(v)

def _read_csv_robust(uploaded_file) -> pd.DataFrame:
    """Lê CSVs 'bagunçados': detecta encoding/delimitador, trata campos extras, normaliza cabeçalhos e devolve colunas padrão."""
    # 1) bytes -> texto (tentando encodings comuns)
    raw = uploaded_file.getvalue()
    text = None
    for enc in ("utf-8-sig","utf-8","cp1252","latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    # 2) detecta delimitador
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        delimiter = dialect.delimiter
    except Exception:
        counts = {d: sample.count(d) for d in [",",";","\t","|"]}
        delimiter = max(counts, key=counts.get)

    # 3) DictReader com captura de colunas excedentes na linha
    buf = io.StringIO(text)
    reader = csv.DictReader(buf, delimiter=delimiter, quotechar='"', escapechar="\\", restkey="_extra", restval="")

    # 4) normaliza cabeçalhos e mapeia sinônimos
    def norm(s):
        return s.strip().lower().replace(" ", "_") if isinstance(s, str) else s

    mapping = {}
    if reader.fieldnames:
        for h in reader.fieldnames:
            n = norm(h)
            if n in ("id","codigo_profissional","codigo","id_candidato"):
                mapping[h] = "id"
            elif n in ("name","nome","nome_completo","full_name"):
                mapping[h] = "name"
            elif n in ("cv_pt","cv","curriculo","currículo"):
                mapping[h] = "cv_pt"
            elif n in ("competencias","competências","skills"):
                mapping[h] = "competencias"
            elif n in ("observacoes","observações","obs"):
                mapping[h] = "observacoes"
            else:
                mapping[h] = n

    rows = []
    for row in reader:
        d = {mapping.get(k, norm(k)): v for k, v in row.items()}

        idv = d.get("id") or ""
        namev = d.get("name") or d.get("nome") or ""
        cvv = d.get("cv_pt") or ""
        compv = d.get("competencias") or ""
        obsv = d.get("observacoes") or ""

        # junta colunas excedentes ao CV (se houver)
        extra = row.get("_extra", [])
        if isinstance(extra, list) and extra:
            cvv = " ".join([cvv] + [str(x) for x in extra if x is not None]).strip()

        rows.append({
            "id": _s(idv),
            "name": _s(namev),
            "cv_pt": _s(cvv),
            "competencias": _s(compv),
            "observacoes": _s(obsv),
        })

    df = pd.DataFrame(rows, columns=["id","name","cv_pt","competencias","observacoes"]).fillna("")
    # se não tiver id, cria um
    if df["id"].eq("").all():
        df["id"] = df.index.astype(str)
    return df

def _prepare_df(uploaded_csv) -> pd.DataFrame:
    # primeiro tenta o leitor robusto; se falhar totalmente, cai para pandas.read_csv padrão
    try:
        df = _read_csv_robust(uploaded_csv)
        # limpeza final e tipos
        for c in ["id","name","cv_pt","competencias","observacoes"]:
            if c not in df.columns: df[c] = ""
        df[["id","name","cv_pt","competencias","observacoes"]] = df[["id","name","cv_pt","competencias","observacoes"]].fillna("").astype(str)
        return df
    except Exception as e:
        st.warning(f"Leitura robusta falhou ({e}). Tentando leitura padrão...")
        df = pd.read_csv(uploaded_csv).fillna("")
        # normaliza mínimos
        need = ["id","name","cv_pt","competencias","observacoes"]
        for c in need:
            if c not in df.columns: df[c] = ""
        return df[need].astype(str)

def _build_payload(df: pd.DataFrame) -> dict:
    return {
        "titulo_vaga": _s(titulo),
        "principais_atividades": _s(atividades),
        "competencias": _s(comp_job),
        "observacoes": _s(obs_job),
        "k": int(k),
        "use_threshold": bool(use_threshold),
        "candidates": [
            {
                "id": _s(r["id"]),
                "name": _s(r["name"]),
                "cv_pt": _s(r["cv_pt"]),
                "competencias": _s(r["competencias"]),
                "observacoes": _s(r["observacoes"]),
            }
            for _, r in df.iterrows()
        ],
    }

# ---------------- Main: File upload + Run button ----------------
st.subheader("Candidatos (CSV com colunas: id, name, cv_pt, competencias, observacoes)")
up = st.file_uploader("Carregue um CSV", type=["csv"], key="file_uploader")

# Persistência na sessão
for key, default in [("cand_df", None), ("result_df", None), ("last_threshold", None), ("last_used_k", None)]:
    if key not in st.session_state: st.session_state[key] = default

# Atualiza DataFrame da sessão ao subir novo arquivo
if up is not None:
    try:
        st.session_state["cand_df"] = _prepare_df(up)
        st.info("Arquivo carregado. Clique em **Rodar ranking** para processar com as configurações atuais.")
    except Exception as e:
        st.error(f"Falha ao ler CSV: {e}")

# Preview do CSV atual
if st.session_state["cand_df"] is not None:
    st.caption("Prévia dos candidatos (primeiras linhas):")
    st.dataframe(st.session_state["cand_df"].head(10), use_container_width=True)
else:
    st.info("Carregue um CSV para habilitar o ranking.")

# Botões
colA, colB = st.columns([1,1])
run_clicked = colA.button("Rodar ranking", type="primary", disabled=st.session_state["cand_df"] is None)
clear_clicked = colB.button("Limpar resultados")

if clear_clicked:
    st.session_state["result_df"] = None
    st.session_state["last_threshold"] = None
    st.session_state["last_used_k"] = None

if run_clicked and st.session_state["cand_df"] is not None:
    payload = _build_payload(st.session_state["cand_df"])
    try:
        r = requests.post(API, headers={"Content-Type":"application/json; charset=utf-8"}, data=json.dumps(payload))
        if r.ok:
            out = r.json()
            items = pd.DataFrame(out.get("items", []))
            st.session_state["last_threshold"] = out.get("threshold_used", None)
            st.session_state["last_used_k"] = out.get("used_k", None)

            if items.empty:
                st.session_state["result_df"] = None
                st.warning("Nenhum candidato retornado. Tente desmarcar 'Aplicar threshold' ou ajuste o contexto da vaga.")
            else:
                base = st.session_state["cand_df"][["id","name"]].drop_duplicates("id")
                if "name" not in items.columns:
                    items["name"] = ""
                items = items.merge(base, how="left", on="id", suffixes=("_api", ""))
                if "name_api" in items.columns:
                    items["name"] = items.apply(lambda r: r["name_api"] if r.get("name_api") not in [None, "", "nan"] else r["name"], axis=1)
                    items = items.drop(columns=["name_api"])
                show = items[["name","id","score","pass_by_threshold"]].sort_values("score", ascending=False)
                show["score"] = show["score"].map(lambda x: round(float(x), 4))
                st.session_state["result_df"] = show
        else:
            st.session_state["result_df"] = None
            st.error(f"Erro {r.status_code}: {r.text}")
    except Exception as e:
        st.session_state["result_df"] = None
        st.error(f"Falha ao chamar API: {e}")

# Render do resultado
if st.session_state["result_df"] is not None:
    thr = st.session_state["last_threshold"]
    used_k = st.session_state["last_used_k"]
    if thr is not None and used_k is not None:
        st.success(f"Threshold usado: {thr:.4f} | itens retornados: {used_k}")
    st.dataframe(st.session_state["result_df"], use_container_width=True)
    csv_bytes = st.session_state["result_df"].to_csv(index=False).encode("utf-8")
    st.download_button("Baixar ranking (CSV)", data=csv_bytes, file_name="ranking.csv", mime="text/csv")
else:
    st.caption("Sem resultados no momento. Carregue um CSV e clique em **Rodar ranking**.")
