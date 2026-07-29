# End-to-End MLOps Machine Learning Pipeline

An end-to-end **MLOps project** demonstrating how to build, validate, test, containerize, and deploy a machine learning model using modern MLOps practices.

The primary objective of this project is to show how a production-ready ML system is built—from data validation to model inference—with automation, testing, Docker, CI/CD, and experiment tracking.

---

# Project Overview

This project implements a complete machine learning workflow that includes:

* Data validation
* Data preprocessing
* Model inference
* REST API using FastAPI
* Automated testing with Pytest
* Docker containerization
* GitHub Actions CI pipeline
* Experiment tracking
* Reproducible project structure

Instead of focusing only on model training, this repository emphasizes **building reliable and maintainable ML systems**.

---

# MLOps Workflow

```
                Raw Data
                    │
                    ▼
         Data Validation
                    │
                    ▼
          Data Preprocessing
                    │
                    ▼
          Trained ML Model
                    │
                    ▼
             FastAPI Service
                    │
                    ▼
            Prediction API
                    │
                    ▼
          Docker Container
                    │
                    ▼
        CI Pipeline (GitHub Actions)
```

---

# Project Structure

```
.
├── app/
│   ├── main.py
│   ├── api/
│   ├── services/
│   └── utils/
│
├── artifacts/
│   ├── models/
│   └── reports/
│
├── config/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Features

* Data validation before inference
* Data preprocessing pipeline
* Machine learning prediction service
* REST API built with FastAPI
* Automated unit testing
* Docker support
* CI using GitHub Actions
* Experiment tracking
* Modular project structure
* Production-ready code organization

---

# Tech Stack

### Machine Learning

* Python
* Scikit-learn
* CatBoost
* Pandas
* NumPy

### API

* FastAPI
* Uvicorn

### Testing

* Pytest

### MLOps

* Docker
* GitHub Actions
* Experiment Tracking (e.g., Weights & Biases)
* Git
* DVC

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git

cd <repository-name>
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the API

```bash
uvicorn app.main:app --reload
```

Open

```
http://127.0.0.1:8000
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

# Running Tests

Execute all tests

```bash
pytest
```

Generate a coverage report

```bash
pytest --cov=app
```

---

# Docker

Build the Docker image

```bash
docker build -t mlops-project .
```

Run the container

```bash
docker run -p 8000:8000 mlops-project
```

Using Docker Compose

```bash
docker compose up --build
```

---

# CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically runs on every push and pull request.

Typical pipeline stages include:

* Checkout repository
* Set up Python
* Install dependencies
* Run unit tests
* Generate test reports
* Validate build

This helps ensure every code change is verified before merging.

---

# Testing Strategy

The project contains automated tests for:

* API endpoints
* Prediction service
* Data validation
* Data preprocessing
* Utility functions

Automated testing helps catch bugs early and improves confidence in deployments.

---

# API Endpoints

## Health Check

```
GET /
```

Response

```json
{
    "status": "healthy",
    "message": "Application is working fine."
}
```

---

## Prediction

```
POST /predict
```

Accepts CSV input and returns model predictions.

---

# MLOps Best Practices Implemented

* Modular project architecture
* Separation of application and model logic
* Configuration-driven design
* Automated testing
* Dockerized application
* CI pipeline with GitHub Actions
* Experiment tracking
* Version-controlled source code
* Reproducible environments
* Clean project structure
* Production-ready REST API

---

# Future Improvements

* Continuous model training
* Model versioning
* Model registry integration
* Kubernetes deployment
* Monitoring and alerting
* Drift detection
* Automated retraining
* Infrastructure as Code (Terraform)
* CD pipeline for cloud deployment

---

# Learning Objectives

This project demonstrates practical MLOps concepts including:

* Building production-ready ML applications
* Serving models through REST APIs
* Writing reliable automated tests
* Containerizing ML applications
* Implementing CI pipelines
* Organizing scalable ML codebases
* Following software engineering best practices for machine learning

---

# License

This project is intended for educational purposes and learning modern MLOps practices.
