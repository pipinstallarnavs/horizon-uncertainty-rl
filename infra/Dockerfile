FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY horizon ./horizon
RUN pip install --no-cache-dir -e ".[serve]"

# The service account key is mounted at runtime from Secret Manager into a
# tmpfs; it is never baked into the image and never enters the repository.
ENV HORIZON_SA_KEY=/run/secrets/sa-key.json

CMD ["uvicorn", "horizon.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
