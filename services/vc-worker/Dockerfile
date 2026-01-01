FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set PYTHONPATH so Python can find shared and services modules
ENV PYTHONPATH=/app

EXPOSE 8003

CMD ["uvicorn", "services.vc-worker.main:app", "--host", "0.0.0.0", "--port", "8003"]

