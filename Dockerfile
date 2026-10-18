# Official Python 3.13 / Debian Bookworm image, pinned on 2026-10-18.
# The dependency lock targets Linux amd64 / CPython 3.13.
FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SUPPLYBEACON_MASTER_WRITE_ENABLED=false \
    SUPPLYBEACON_LLM_ENABLED=false \
    LANGSMITH_TRACING=false \
    LANGCHAIN_TRACING_V2=false \
    SUPPLYBEACON_ALLOWED_ORIGINS=https://supply-beacon.vercel.app \
    SUPPLYBEACON_CURRENT_DATA_DIR=/app/data/persistent/current \
    SUPPLYBEACON_MASTER_DB_PATH=/app/data/persistent/master/supplybeacon_master.db \
    SUPPLYBEACON_MASTER_GENERATED_DIR=/app/data/persistent/master/generated

WORKDIR /app
COPY requirements-api.lock .
RUN python -m pip install --no-cache-dir --no-compile --only-binary=:all: --require-hashes -r requirements-api.lock \
    && python -m pip check

COPY src ./src
COPY data/graph/supply_chain_graph.graphml ./data/graph/supply_chain_graph.graphml
COPY data/graph/nodes.csv ./data/graph/nodes.csv
COPY data/graph/edges.csv ./data/graph/edges.csv
COPY data/gnn_prediction/node_embeddings.csv ./data/gnn_prediction/node_embeddings.csv
COPY data/processed/brand_supplier_mapping.csv ./data/processed/brand_supplier_mapping.csv
COPY data/processed/electronics_supplychain_dataset.csv ./data/processed/electronics_supplychain_dataset.csv
COPY data/risk_prediction/risk_scores.csv ./data/risk_prediction/risk_scores.csv
COPY deploy/seed/current ./deploy/seed/current
COPY deploy/seed/analysis/dataco/modeling ./deploy/seed/analysis/dataco/modeling
COPY deploy/seed-manifest.json ./deploy/seed-manifest.json
COPY deploy/audit_seed.py ./deploy/audit_seed.py
COPY deploy/start.py ./deploy/start.py
RUN python deploy/audit_seed.py

EXPOSE 8000
CMD ["python", "deploy/start.py"]
