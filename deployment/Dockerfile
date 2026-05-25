FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/app/ .

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MODEL_URI=models:/vehicle-type-classifier@champion

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
