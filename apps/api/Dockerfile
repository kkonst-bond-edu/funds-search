# Multi-stage build for API service to reduce image size
# Stage 1: Build dependencies (if needed for compilation)
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies only if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements/ requirements/
RUN pip install --no-cache-dir --user -r requirements/api.txt

# Stage 2: Runtime image (minimal)
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PYTHONPATH so Python can find shared and apps modules
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Copy application code (including src directory for vacancy search)
COPY . .
# Ensure src directory is accessible
RUN ls -la /app/src 2>/dev/null || echo "Warning: src directory not found"

EXPOSE 8000

# Health check for Azure Container Apps
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
