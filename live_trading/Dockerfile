FROM python:3.12-slim

WORKDIR /app

# Install system deps for numpy/pandas compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY csp/ csp/

# Ensure print() output appears immediately in Cloud Logging
ENV PYTHONUNBUFFERED=1

# Entry point
CMD ["python", "-m", "csp"]
