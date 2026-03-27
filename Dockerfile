FROM python:3.14-slim

WORKDIR /app

# Install uv for faster dependency management
RUN pip install --no-cache-dir uv

# Copy dependencies first for better caching
COPY pyproject.toml /app/pyproject.toml

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY src/ /app/src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command can be overridden
# CMD ["python", "main_99.py"]
