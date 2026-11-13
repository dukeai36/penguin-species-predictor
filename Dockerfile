
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the API and configuration
COPY api/ ./api/
COPY config.yaml .

# Copy the trained model
COPY models/penguin_model.joblib ./models/penguin_model.joblib

# Configure the API port
ENV PORT=8080

EXPOSE 8080

# Start FastAPI
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
