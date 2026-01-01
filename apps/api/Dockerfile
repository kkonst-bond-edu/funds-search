FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set PYTHONPATH so Python can find shared and apps modules
ENV PYTHONPATH=/app

# Copy minimal API requirements (excludes torch, transformers, docling)
COPY apps/api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY . .

EXPOSE 8000

# Health check for Azure Container Apps
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

