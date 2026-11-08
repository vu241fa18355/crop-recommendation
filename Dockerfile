FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY deployment/model.json deployment/model_metadata.json ./deployment/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]