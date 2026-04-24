FROM python:3.11-slim

WORKDIR /app

COPY requirements.docker.txt .

RUN pip install --no-cache-dir --timeout=120 -r requirements.docker.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]