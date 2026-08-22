FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install UV
RUN pip install --no-cache-dir uv

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src ./src
COPY utils ./utils
COPY config ./config
COPY entity ./entity

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uv","run","uvicorn","src.api.app:app","--host","0.0.0.0","--port","8000"]