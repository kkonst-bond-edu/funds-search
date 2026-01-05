FROM python:3.10-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Обновляем pip и устанавливаем uvicorn напрямую
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir uvicorn[standard]

# Копируем зависимости сервиса
COPY services/cv-processor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Docling models during build by processing a dummy PDF
# This forces Docling to download ALL necessary models (OCR, Layout, etc.) into the image
# Creating a minimal valid PDF for testing
RUN python -c "from docling.document_converter import DocumentConverter; from pathlib import Path; p = Path('dummy.pdf'); p.write_bytes(b'%PDF-1.0\n1 0 obj<</Type/Page/Parent 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[1 0 R]>>endobj\n3 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\ntrailer<</Root 3 0 R>>%%EOF'); DocumentConverter().convert(p); p.unlink(); print('All models pre-downloaded successfully')" || echo "Note: Some models may download on first use"

# Копируем весь проект
COPY . .

# Set PYTHONPATH so Python can find shared and services modules
ENV PYTHONPATH=/app

EXPOSE 8001

# ГАРАНТИРОВАННЫЙ способ запуска через модуль
CMD ["python", "-m", "uvicorn", "services.cv-processor.main:app", "--host", "0.0.0.0", "--port", "8001"]