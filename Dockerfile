FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Create runtime directories
RUN mkdir -p /app/artifacts/logs

# Application code
COPY src ./src
COPY utils ./utils
COPY config ./config
COPY entity ./entity

# Model preprocessing artifact
COPY artifacts/encoder.joblib ./artifacts/encoder.joblib

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]