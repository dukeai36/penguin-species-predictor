
FROM python:3.13-slim

WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY config.yaml .

# Copy locally trained model for serving
COPY models/penguin_model.joblib ./models/penguin_model.joblib

# Configure server port
ENV PORT=8080

EXPOSE 8080

# Default command serves predictions
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
