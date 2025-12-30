FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Копируем requirements ИМЕННО из папки сервиса
# Мы сейчас находимся в корне проекта (контекст сборки)
COPY services/embedding-service/requirements.txt ./requirements.txt

# Устанавливаем легкий torch для CPU (чтобы билд не падал по памяти)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic>=2.0.0 \
    transformers>=4.44.0 \
    torch \
    sentencepiece

# 2. Предзагрузка модели
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-m3'); AutoModel.from_pretrained('BAAI/bge-m3')"

# 3. Копируем всё остальное
COPY . .

EXPOSE 8000

CMD ["uvicorn", "services.embedding-service.main:app", "--host", "0.0.0.0", "--port", "8000"]