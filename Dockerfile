FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python -m src.data_pipeline.ingest --bank-sample-size 5000 --complaints-sample-size 5000 && python -m src.data_pipeline.validate && python -m src.data_pipeline.features && python -m src.training.train && python -m src.training.evaluate && python -m src.rag.build_index --max-features 5000 && uvicorn src.serving.serve:app --host 0.0.0.0 --port 8000"]