# Docker Documentation

## Overview

The project includes a Dockerfile for packaging the Student Success Prediction API together with its Python dependencies, source code, configuration, and model artifacts.

The container runs the FastAPI application using Uvicorn.

```text
Docker Image
    |
    +-- Python 3.11
    +-- System dependencies
    +-- Python dependencies
    +-- app/
    +-- src/
    +-- utils/
    +-- config/
    +-- artifacts/
    +-- entity/
            |
            v
      Uvicorn :8000
            |
            v
 Student Success Prediction API
```

---

# Dockerfile

The Dockerfile uses:

```dockerfile
FROM python:3.11-slim
```

This provides a lightweight Python 3.11 runtime.

The container also sets:

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
```

These settings:

- Prevent Python `.pyc` files from being written.
- Ensure Python logs are written immediately to the container output.

---

# System Dependency

The Docker image installs:

```text
libgomp1
```

This is required by native machine-learning components that depend on the GNU OpenMP runtime.

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

The package list is cleaned afterward to reduce image size.

---

# Working Directory

The application runs from:

```text
/app
```

```dockerfile
WORKDIR /app
```

---

# Installing Python Dependencies

The Dockerfile first copies the dependency file:

```dockerfile
COPY requirements.txt .
```

Then installs the dependencies:

```dockerfile
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
```

The project pins the main runtime dependencies, including:

```text
fastapi==0.116.1
uvicorn[standard]==0.35.0
pandas==2.3.1
numpy==2.3.1
scikit-learn==1.7.0
joblib==1.5.1
catboost==1.2.8
pydantic==2.11.7
python-multipart==0.0.20
PyYAML==6.0.2
```

---

# Copying Application Files

The Docker image copies the following project components:

```dockerfile
COPY app/ ./app/
COPY src/ ./src/
COPY utils/ ./utils/
COPY config/ ./config/
COPY artifacts/ ./artifacts/
COPY entity/ ./entity/
```

This is important because the API imports modules from `app`, `src`, and `utils`, while the inference service requires configuration and model artifacts.

---

# Non-Root Container User

The Dockerfile creates a dedicated user:

```dockerfile
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser
```

The application therefore does not run as the root user.

This is a useful security practice for production containers.

---

# Exposed Port

The API listens on port:

```text
8000
```

The Dockerfile declares:

```dockerfile
EXPOSE 8000
```

`EXPOSE` documents the intended container port. The host-to-container mapping is specified when running the container.

---

# Container Startup

The container starts:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The important part is:

```text
app.main:app
```

This means:

```text
app/
└── main.py
    └── app = create_app()
```

Uvicorn imports the `app` object from `app.main`.

The server binds to:

```text
0.0.0.0:8000
```

so it is accessible outside the container when port `8000` is published.

---

# Build the Docker Image

From the project root:

```bash
docker build -t student-success-api .
```

The project root should contain:

```text
Dockerfile
requirements.txt
app/
src/
utils/
config/
artifacts/
entity/
```

---

# Run the Container

Run:

```bash
docker run --name student-success-api -p 8000:8000 student-success-api
```

The mapping:

```text
8000:8000
```

means:

```text
Host port 8000 → Container port 8000
```

The API is then available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# Test the Container

## Health Check

```bash
curl http://localhost:8000/
```

Expected response:

```json
{
  "status": "healthy",
  "message": "Application is working fine."
}
```

## Prediction

From the project directory:

```bash
curl -X POST \
  -F "file=@test_sample.csv" \
  http://localhost:8000/predict/file
```

---

# Using Docker Desktop

After running the container:

1. Open Docker Desktop.
2. Find `student-success-api`.
3. Check the container logs.
4. Open `http://localhost:8000/docs`.
5. Use Swagger UI to test the prediction endpoint.

---

# Useful Docker Commands

## List running containers

```bash
docker ps
```

## List all containers

```bash
docker ps -a
```

## View API logs

```bash
docker logs student-success-api
```

Follow logs continuously:

```bash
docker logs -f student-success-api
```

## Stop the container

```bash
docker stop student-success-api
```

## Start an existing container

```bash
docker start student-success-api
```

## Remove the container

```bash
docker rm student-success-api
```

## Force remove a running container

```bash
docker rm -f student-success-api
```

## List images

```bash
docker images
```

## Remove the image

```bash
docker rmi student-success-api
```

---

# Rebuild After Code Changes

If the application code or dependencies change, rebuild the image:

```bash
docker build -t student-success-api .
```

If a container with the same name already exists:

```bash
docker rm -f student-success-api
```

Then run the new image:

```bash
docker run --name student-success-api -p 8000:8000 student-success-api
```

---

# Troubleshooting

## Container name already in use

Error:

```text
Conflict. The container name "/student-success-api" is already in use
```

Check existing containers:

```bash
docker ps -a
```

Remove the old container:

```bash
docker rm -f student-success-api
```

Then run the new container.

---

## `Attribute "app" not found`

If Uvicorn reports that `app` cannot be found, verify the module path.

For this project the application is located at:

```text
app/main.py
```

and the Dockerfile correctly starts:

```text
app.main:app
```

Do not use:

```text
main:app
```

unless the FastAPI application object is actually defined in the root `main.py`.

---

## `libgomp.so.1` not found

If the container reports:

```text
libgomp.so.1: cannot open shared object file
```

the OpenMP runtime is missing.

The Dockerfile addresses this by installing:

```text
libgomp1
```

If the error appears, rebuild the image rather than only restarting the old container:

```bash
docker build --no-cache -t student-success-api .
```

---

## API is not accessible

Check whether the container is running:

```bash
docker ps
```

Check the logs:

```bash
docker logs student-success-api
```

Verify that port mapping contains:

```text
0.0.0.0:8000->8000/tcp
```

Then open:

```text
http://localhost:8000/
```

---

## Prediction fails because of input columns

The API validates uploaded columns against:

```text
artifacts/feature_names.joblib
```

Therefore, inspect the expected feature artifact and make sure the uploaded CSV matches the trained feature set.

The API rejects both:

- Missing columns
- Unexpected columns

---

# `.dockerignore`

The project contains a `.dockerignore` file. It should be used to prevent unnecessary files from being sent to the Docker build context.

Typical candidates include:

```text
.venv/
__pycache__/
.pytest_cache/
.git/
.github/
wandb/
notebooks/
remote storage/
*.pyc
```

Do not exclude runtime files that the Dockerfile explicitly needs, especially:

```text
app/
src/
utils/
config/
artifacts/
entity/
requirements.txt
```

---

# Production Notes

The current Dockerfile already includes several good container practices:

- Uses a slim Python base image.
- Installs only required system packages.
- Removes apt package lists.
- Uses `pip --no-cache-dir`.
- Runs as a non-root user.
- Binds Uvicorn to `0.0.0.0`.
- Includes model artifacts inside the image.

For a production deployment, artifact management can later be separated from the application image so that model versions can be updated without rebuilding the complete API image.

---

# Complete Workflow

```bash
# Build
docker build -t student-success-api .

# Run
docker run --name student-success-api -p 8000:8000 student-success-api

# Check
curl http://localhost:8000/

# Predict
curl -X POST \
  -F "file=@test_sample.csv" \
  http://localhost:8000/predict/file

# View logs
docker logs -f student-success-api
```

The service is then available at:

```text
http://localhost:8000
```

with interactive API documentation at:

```text
http://localhost:8000/docs
```
