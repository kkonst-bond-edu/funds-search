FROM python:3.10-slim

# Установка системных зависимостей для sentencepiece и numpy
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY services/embedding-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка модели
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-m3'); AutoModel.from_pretrained('BAAI/bge-m3')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]