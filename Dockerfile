
FROM python:3.13-slim

WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Use the public dataset without Google Cloud credentials
ENV PENGUIN_DATA_SOURCE=public

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY config.yaml .

# Train the model inside Docker
RUN python -m src.train

# Configure server port
ENV PORT=8080

EXPOSE 8080

# Start the prediction API
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
