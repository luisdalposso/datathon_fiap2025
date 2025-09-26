# Decision Match — Pipeline, API, UI & Observability

Sistema de **matching candidato ↔ vaga** com pipeline de ML, **API em FastAPI**, **UI em Streamlit** e **observabilidade** (Prometheus + Grafana + Drift/Evidently). Projeto pronto para desenvolvimento local (Windows 11) e para execução via **Docker Compose**.

---

## Sumário
- [Arquitetura](#arquitetura)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Pré‑requisitos](#pré-requisitos)
- [Instalação e desenvolvimento local (Windows 11 + VS Code)](#instalação-e-desenvolvimento-local-windows-11--vs-code)
- [Treino do modelo](#treino-do-modelo)
- [Testes e cobertura](#testes-e-cobertura)
- [Subir a API e a UI localmente](#subir-a-api-e-a-ui-localmente)
- [Docker: build e subida dos serviços](#docker-build-e-subida-dos-serviços)
- [URLs e endpoints](#urls-e-endpoints)
- [Observabilidade e Drift](#observabilidade-e-drift)
- [Provisionamento automático do Grafana (as‑code)](#provisionamento-automático-do-grafana-as-code)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Troubleshooting (Windows/Docker/Grafana)](#troubleshooting-windowsdockergrafana)
- [Boas práticas e próximos passos](#boas-práticas-e-próximos-passos)
- [Licença](#licença)

---

## Arquitetura

- **Pipeline de ML** (TF‑IDF + modelo linear scikit‑learn) → artefatos versionados em `models/artifacts/`.
- **API (FastAPI)** expõe `POST /score`, `POST /rank-candidates`, métricas Prometheus em `/metrics` e health em `/health`.
- **UI (Streamlit)** para testar a API com inputs reais.
- **Drift Service** (FastAPI + Evidently): lê `monitoring/requests_log.csv`, compara com `models/artifacts/baseline_features.csv` e publica:
  - Métricas em `/metrics` (Prometheus).
  - Relatórios HTML em `monitoring/drift_reports/`.
- **Prometheus** coleta métricas de `api:8000` e `drift:8001`.
- **Grafana** oferece dashboards provisionados por código (datasource + painel “Decision Match — Observability”).

---

## Estrutura do repositório

```
├─ docker/                  # Dockerfiles (api, ui, drift)
├─ monitoring/
│  ├─ grafana/
│  │  ├─ provisioning/
│  │  │  ├─ datasources/datasource.yml
│  │  │  └─ dashboards/dashboards.yml + dm_observability.json
│  │  └─ ... (dados persistidos do Grafana)
│  ├─ prometheus.yml        # Targets api/drift
│  ├─ drift_reports/        # Relatórios Evidently (gerados em runtime)
│  └─ requests_log.csv      # Log de tráfego (gerado pela API)
├─ models/
│  └─ artifacts/            # Artefatos do modelo (baseline, vetores, etc.)
├─ scripts/                 # *.bat para Windows (setup, test, serve, train, docker)
├─ src/
│  ├─ api/                  # FastAPI (main, schemas)
│  ├─ features/             # Limpeza/feature engineering
│  ├─ labeling/             # Mapeamento de targets
│  ├─ monitoring/           # drift_service (Evidently)
│  └─ ...                   # utils, pipeline, treino, etc.
├─ tests/                   # unit + integration (pytest)
├─ ui/                      # Streamlit app
├─ docker-compose.yml
├─ pyproject.toml | requirements.txt
└─ README.md
```

> Observação: as pastas de `monitoring` e `models/artifacts` são montadas como volumes pelo Docker Compose.

---

## Pré‑requisitos

- **Windows 11** + **Docker Desktop** (WSL2) + **Git** + **VS Code**.
- Python **3.11** (opcional para rodar local sem Docker).

---

## Instalação e desenvolvimento local (Windows 11 + VS Code)

```bat
:: 1) criar ambiente e instalar deps
scripts\setup_venv.bat

:: 2) (opcional) ativar venv manualmente
call .venv\Scripts\activate
```

---

## Treino do modelo

Gera artefatos em `models/artifacts/` e baseline para drift (`baseline_features.csv`).

```bat
scripts\train.bat
```

Ao final, são exibidas métricas de validação cruzada (ex.: **NDCG@5**, **F1_mean**, **ROC_AUC_mean**) e é definido um `threshold_topk` (utilizado pela API).

---

## Testes e cobertura

Executa **pytest** + **pytest-cov** e valida o mapeamento de targets e limpeza de texto, além de integrar a API.

```bat
scripts\test.bat
```

Exemplo de cobertura obtida no projeto: **96%** (com `src/api/main.py` chegando a ~94%).

---

## Subir a API e a UI localmente

```bat
:: API (FastAPI)
scripts\serve.bat
:: abre: http://localhost:8000/docs  e  http://localhost:8000/health

:: UI (Streamlit) — se houver script específico, usar:
:: streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0
```

---

## Docker: build e subida dos serviços

Build das imagens e subida dos containers (API, UI, Drift, Prometheus, Grafana):

```bat
docker compose build
docker compose up -d
docker compose ps
```

> Se preferir subir serviços isolados: `docker compose up -d api ui`, etc.

Rede única `dm_net` conecta todos os serviços. Volumes:
- `./models/artifacts:/app/models/artifacts:ro`
- `./monitoring:/monitoring`

### Parar e remover
```bat
docker compose down
```

---

## URLs e endpoints

**UI (Streamlit):**  
- `http://localhost:8501` — interface para interagir com a API.

**API (FastAPI):**  
- `http://localhost:8000` — raiz  
- `http://localhost:8000/health` — healthcheck (modelo, threshold, etc.)  
- `http://localhost:8000/docs` — Swagger UI  
- `http://localhost:8000/metrics` — métricas Prometheus  
- `POST http://localhost:8000/score` — score de um candidato  
- `POST http://localhost:8000/score-batch` — score em lote  
- `POST http://localhost:8000/rank-candidates` — ranking

**Drift Service:**  
- `http://localhost:8001/health` — status (baseline/log)  
- `http://localhost:8001/metrics` — métricas Prometheus (p-value, flag)  
- Relatórios Evidently: `monitoring/drift_reports/` (gerados em runtime)

**Prometheus:**  
- `http://localhost:9090` — console  
- `http://localhost:9090/targets` — alvos de scrape (api/drift)

**Grafana:**  
- `http://localhost:3000` — dashboards (login **admin/admin**)  
- Dashboard provisionado: **Decision Match — Observability** (pasta *Decision Match*)

---

## Observabilidade e Drift

- A API expõe contadores/histogramas: **requests total por endpoint/status** e **latência** (*histogram buckets*) → usados para **Requests/s**, **Error rate** e **p95** no Grafana.
- O Drift Service roda a cada **60s**:
  1. Carrega baseline (`models/artifacts/baseline_features.csv`).
  2. Lê a janela recente de produção (`monitoring/requests_log.csv`), gerada pelas rotas da API.
  3. Executa Evidently (**DataDriftPreset**), exporta:
     - **Métricas**: `dm_drift_p_value{feature}`, `dm_drift_detected{feature}`.
     - **Relatório HTML**: `monitoring/drift_reports/drift_<timestamp>.html`.

> Para ver dados no painel, gere tráfego (UI ou `POST /score`) e aguarde ~1–2 min.

### Exemplo para popular rapidamente (PowerShell)

```powershell
for ($i=0; $i -lt 250; $i++) {
  $body = @{
    cv_pt="Dev Python com FastAPI e Docker $i"
    titulo_vaga="Backend Python"
    principais_atividades="APIs REST"
    competencias="Python; FastAPI; Docker; SQL"
    observacoes=""
  } | ConvertTo-Json
  Invoke-RestMethod -Method POST -Uri http://localhost:8000/score -Body $body -ContentType "application/json" | Out-Null
}
```

---

## Provisionamento automático do Grafana (as‑code)

Já incluso em `monitoring/grafana/provisioning/`:

- **Datasource**: Prometheus (URL interna `http://prometheus:9090`, `uid: PROM`).
- **Dashboards**: provider aponta para `/etc/grafana/provisioning/dashboards` e carrega `dm_observability.json` (painéis: Requests/s, Error rate, p95, Drift p‑value, Drift flag).

Para re‑aplicar:
```bat
docker compose restart grafana
docker compose logs -f grafana
```

---

## Variáveis de ambiente

| Nome                | Serviço | Default | Descrição |
|---------------------|--------:|:------:|-----------|
| `THRESHOLD_TOPK`    | API     | `0.5`  | Limite mínimo de score para considerar um candidato |
| `TARGET_K`          | API     | `5`    | Top‑K retornado pelo ranking |
| `MONITORING_DIR`    | API/Drift | `/monitoring` | Pasta compartilhada para logs/relatórios |
| `DRIFT_REPORTS_DIR` | Drift   | (opcional) | Se definido, sobrescreve o diretório de relatórios (por padrão `MONITORING_DIR/drift_reports`) |

---

## Troubleshooting (Windows/Docker/Grafana)

- **Docker API/Context no Windows:** se vir `request returned 500 Internal Server Error` ou problemas no contexto `desktop-linux`, alterne com:
  ```powershell
  docker context ls
  docker context use desktop-linux
  ```
- **`drift` reiniciando com `Read-only file system`:** não grave relatórios em `/app/models/artifacts` (volume RO). Use `/monitoring/drift_reports` (já configurado).
- **Prometheus `no such host drift`:** garanta que todos os serviços estão na rede `dm_net` no `docker-compose.yml` e reinicie a stack.
- **Grafana não carrega dashboard JSON:** salve o arquivo **sem BOM** (UTF‑8). No VS Code: “Save with Encoding → UTF‑8”. Reinicie o Grafana.
- **Dashboards “No data”:** selecione **All** nas variáveis, ajuste time range (ex.: *Last 1 hour*) e gere tráfego (UI ou loop de chamadas).

---

## Boas práticas e próximos passos

- **SLOs**: defina metas claras (ex.: *p95 ≤ 1s*, *error rate < 1%*). Configure **alertas** no Grafana (p95, erro, drift).  
- **Dados/Privacidade**: evite logar PII; use proxies de métricas (comprimentos, scores agregados).  
- **MLOps**: versionar artefatos do modelo, re‑treino automatizado ao detectar drift, testes de regressão.  
- **CI/CD**: pipelines com lint, testes, build de imagens e deploy com version tags.  

---

## Licença

Projeto acadêmico/educacional. Ajuste a licença conforme política da equipe/organização.
