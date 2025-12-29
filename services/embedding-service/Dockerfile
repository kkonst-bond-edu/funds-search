FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from project root
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/embedding-service/ /app/services/embedding-service/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Run the service
CMD ["uvicorn", "services.embedding-service.main:app", "--host", "0.0.0.0", "--port", "8001"]
