FROM python:3.10-slim

# Install system dependencies required for Docling
# libgomp1 is needed for OpenMP support
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set PYTHONPATH to include app root for imports
ENV PYTHONPATH=/app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/ /app/services/
COPY shared/ /app/shared/

EXPOSE 8002

# Default command
CMD ["uvicorn", "services.cv-processor.main:app", "--host", "0.0.0.0", "--port", "8002"]

