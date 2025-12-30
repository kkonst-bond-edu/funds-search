FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Копируем файл зависимостей
COPY services/embedding-service/requirements.txt .

# 2. Устанавливаем зависимости из файла (флаг --extra-index-url внутри файла сделает всё сам)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Предзагрузка модели
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-m3'); AutoModel.from_pretrained('BAAI/bge-m3')"

# 4. Копируем код
COPY . .

EXPOSE 8000

CMD ["uvicorn", "services.embedding-service.main:app", "--host", "0.0.0.0", "--port", "8000"]