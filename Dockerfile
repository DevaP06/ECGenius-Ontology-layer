FROM python:3.12-slim

WORKDIR /app

# Install deps before copying source so this layer is cached across rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs at runtime
COPY api.py .
COPY inference/   inference/
COPY rules_engine/ rules_engine/
COPY history_module/ history_module/
COPY ontology/    ontology/

EXPOSE 8002

# ALLOWED_ORIGINS — comma-separated list of origins the frontend runs on.
# Override at `docker run -e` or in docker-compose.yml.
ENV ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]
