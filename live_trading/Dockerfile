# CSP Strategy - Cloud Run Job
# Python 3.12 slim (3.14 not yet in official Docker images)

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package (notebooks excluded via .dockerignore)
COPY csp/ ./csp/

# Set Python path so we can run as module
ENV PYTHONPATH=/app

# Default: run the strategy loop
# Override with: docker run ... python -m csp.main --test-orders
ENTRYPOINT ["python", "-m", "csp.main", "--run"]
