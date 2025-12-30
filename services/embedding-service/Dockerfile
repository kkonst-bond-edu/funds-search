FROM python:3.10-slim

# Системные зависимости
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Копируем ТОЛЬКО requirements этого сервиса
COPY services/embedding-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. Предзагрузка модели (теперь transformers 4.44.0, конфликта не будет)
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-m3'); AutoModel.from_pretrained('BAAI/bge-m3')"

# 3. Копируем остальной код
COPY . .

EXPOSE 8000

# Запуск. Путь должен быть от корня, так как мы копируем весь проект
CMD ["uvicorn", "services.embedding-service.main:app", "--host", "0.0.0.0", "--port", "8000"]