# Decision Match — Pipeline + API (Baseline scikit-learn)

Sistema de matching candidato↔vaga com pipeline de ML, API em FastAPI, Docker e testes.
**Baseline 100% scikit-learn (TF‑IDF + modelo linear)**, sem PyTorch (compatível com Windows 11).

## Como começar (Windows 11 + VS Code)

1) Criar venv e instalar dependências:
```bat
scripts\setup_venv.bat
```

2) Treinar o modelo (usa data/raw/*.json):
```bat
scripts\train.bat
```

3) Subir a API:
```bat
scripts\serve.bat
```
Abra: http://127.0.0.1:8000/docs

4) Rodar testes + cobertura:
```bat
scripts\test.bat
```

5) Docker (build + run):
```bat
scripts\build_run_docker.bat
```

## Estrutura
Veja `src/` para módulos (config, data, features, labeling, modeling, api, monitoring, utils).
Artefatos do modelo ficam em `models/artifacts/`.
Relatórios de métricas/drift em `models/reports/`.
