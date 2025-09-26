# docker/drift.Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# utilitário opcional (curl) para troubleshooting dentro do container
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependências base do projeto
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    # libs específicas do serviço de drift
    && pip install --no-cache-dir evidently==0.4.33 fastapi uvicorn

# Código-fonte
COPY . .

# Diretório montado via docker-compose para logs/monitoramento
ENV MONITORING_DIR=/monitoring

EXPOSE 8001

# /metrics e /health são servidos pelo uvicorn; NÃO usar start_http_server no app
CMD ["uvicorn", "src.monitoring.drift_service:app", "--host", "0.0.0.0", "--port", "8001"]
