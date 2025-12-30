FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Копируем только requirements для кеширования слоев
COPY services/embedding-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка модели
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-m3'); AutoModel.from_pretrained('BAAI/bge-m3')"

COPY . .

EXPOSE 8000
# Исправленный путь к приложению
CMD ["uvicorn", "services.embedding-service.main:app", "--host", "0.0.0.0", "--port", "8000"]