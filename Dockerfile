FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY apps/ /app/apps/
COPY services/ /app/services/
COPY shared/ /app/shared/

# Default command (can be overridden in docker-compose)
ENTRYPOINT [ "uvicorn" ]
CMD [ "--host", "0.0.0.0", "apps.api.main:app" ]
