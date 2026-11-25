FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, exported champion model, and source modules
COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
