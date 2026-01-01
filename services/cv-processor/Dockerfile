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

# Копируем весь проект
COPY . .

# Set PYTHONPATH so Python can find shared and services modules
ENV PYTHONPATH=/app

EXPOSE 8001

# ГАРАНТИРОВАННЫЙ способ запуска через модуль
CMD ["python", "-m", "uvicorn", "services.cv-processor.main:app", "--host", "0.0.0.0", "--port", "8001"]