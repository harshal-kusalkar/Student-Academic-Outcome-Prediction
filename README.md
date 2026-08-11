````markdown
# 🎓 Student Academic Outcome Prediction

A modular machine learning project that predicts whether a student is likely to **Dropout**, **Remain Enrolled**, or **Graduate** using academic, demographic, and socioeconomic data.

The project focuses on applying **MLOps architecture and engineering practices** to a machine learning workflow, including data validation, model training and benchmarking, experiment tracking, hyperparameter optimization, reproducibility, testing, and local API deployment.

The raw dataset used for the project was already clean and ready to use. Therefore, data preprocessing and feature engineering were intentionally kept minimal, allowing the project to focus primarily on the **design and implementation of modular MLOps components** rather than extensive data preparation.

---

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Production%20Model-02569B?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)

![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-5C4EE5?style=for-the-badge)

![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

</p>

---

## 🚀 Highlights

- 📊 Multi-class prediction: **Dropout**, **Enrolled**, **Graduate**
- 🧹 Minimal preprocessing because the source dataset was already clean
- 🏗️ Modular and configuration-driven project architecture
- 🤖 Benchmarking of multiple machine learning models
- 🎯 Hyperparameter optimization using **Optuna**
- 📈 Experiment tracking using **Weights & Biases**
- 📦 Reproducible workflows using **DVC**
- 🧪 Automated testing with **Pytest**
- 🔄 Continuous Integration using **GitHub Actions**
- 🚀 FastAPI-based local inference service
- 🐳 Docker-based containerization for the API
- 📚 Separate documentation for architecture, MLOps practices, model benchmarking, API usage, and Docker deployment

> **Implementation scope:** The current project focuses on the machine learning pipeline, MLOps components, and local/containerized inference. Full production cloud deployment and automated CD are planned as future improvements.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Highlights](#-project-highlights)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Documentation](#-documentation)
- [Installation](#-installation)
- [Running the Training Pipeline](#-running-the-training-pipeline)
- [Running the API Locally](#-running-the-api-locally)
- [Using the API](#-using-the-api)
- [Running with Docker](#-running-with-docker)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 📖 Overview

Student Academic Outcome Prediction is a machine learning project for predicting one of three student outcomes:

- **Dropout**
- **Enrolled**
- **Graduate**

The project uses academic, demographic, and socioeconomic information to build a multi-class classification model.

The dataset used in the project was already clean and suitable for modeling. As a result, preprocessing and feature engineering were kept intentionally minimal. The primary engineering focus was placed on building and integrating modular components for model training, evaluation, experiment tracking, versioning, testing, and inference.

The trained model is exposed through a **FastAPI** service that accepts student records as a semicolon-delimited CSV file and returns a prediction for every row.

---

## ✨ Features

### Machine Learning

- Multi-class student outcome prediction
- Benchmarking of multiple machine learning algorithms
- Macro F1-Score used as the primary evaluation metric
- Hyperparameter optimization using **Optuna**
- LightGBM selected as the configured inference model

### MLOps

- Modular and configuration-driven architecture
- Dataset and artifact versioning using **DVC**
- Experiment tracking using **Weights & Biases**
- Automated testing using **Pytest**
- Continuous Integration using **GitHub Actions**

### Inference

- FastAPI-based REST API
- CSV file-based prediction endpoint
- Input feature validation
- Stored model and encoder artifacts
- Interactive Swagger and ReDoc documentation

### Containerization

- Dockerized FastAPI application
- Python 3.11 runtime
- Required system dependency for machine-learning components
- Non-root container user
- Model artifacts included in the Docker image

---

## 🌟 Project Highlights

| Category | Implementation |
|----------|----------------|
| 🏗️ Architecture | Modular, configuration-driven project structure |
| 🤖 Machine Learning | Multi-class classification |
| 📊 Model Development | Benchmarking of multiple ML algorithms |
| 🎯 Optimization | Hyperparameter tuning using **Optuna** |
| 📈 Experiment Tracking | **Weights & Biases (W&B)** |
| 📦 Data Versioning | **DVC** |
| 🚀 API | **FastAPI** |
| 🧪 Testing | **Pytest** |
| 🔄 Continuous Integration | **GitHub Actions** |
| 🐳 Containerization | **Docker** |

The project deliberately keeps preprocessing and feature engineering lightweight because the source dataset was already clean. This allows the implementation to emphasize the organization and integration of MLOps components.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11+ |
| **Machine Learning** | Scikit-learn, LightGBM, XGBoost, CatBoost |
| **Data Processing** | Pandas, NumPy |
| **Hyperparameter Optimization** | Optuna |
| **Experiment Tracking** | Weights & Biases |
| **Data Versioning** | DVC |
| **API Development** | FastAPI, Uvicorn |
| **Model Serialization** | Joblib |
| **Validation** | Custom `ValidateStudentData` |
| **Configuration** | YAML / PyYAML |
| **Testing** | Pytest |
| **CI** | GitHub Actions |
| **Containerization** | Docker |

---

## 📁 Project Structure

```text
Student-Academic-Outcome-Prediction/
│
├── app/
│   └── main.py
│
├── artifacts/
│   ├── models/
│   │   └── best_model.joblib
│   ├── encoder.joblib
│   ├── feature_names.joblib
│   └── ...
│
├── config/
│   └── ...
│
├── data/
│
├── entity/
│
├── src/
│   ├── data/
│   └── inference/
│       ├── loader.py
│       ├── predict.py
│       └── validate.py
│
├── utils/
├── tests/
│
├── dvc.yaml
├── Dockerfile
├── requirements.txt
├── main.py
└── README.md
````

The API runtime uses the application code, source modules, configuration, utilities, and stored model artifacts required for inference.

---

## 📊 Model Performance

Six machine learning algorithms were trained and evaluated using the project's evaluation strategy.

**Macro F1-Score** was selected as the primary metric because the target classes are imbalanced and Macro F1 evaluates performance across the classes without allowing the largest class to dominate the metric.

| Rank | Model        |  Accuracy  |  Macro F1  | Status               |
| :--: | ------------ | :--------: | :--------: | -------------------- |
|  🥇  | **XGBoost**  | **0.7409** | **0.6642** | Finalist             |
|  🥈  | **LightGBM** |   0.7409   |   0.6605   | **Production Model** |
|  🥉  | **CatBoost** |   0.7369   |   0.6520   | Runner-up            |

Although XGBoost achieved the highest Macro F1-Score in the benchmark, **LightGBM** was selected as the configured production/inference model.

The inference configuration uses:

```yaml
model:
  name: LightGBM
  version: v1
```

### Model Artifacts

The inference service uses three main artifacts:

| Artifact               | Purpose                                                  |
| ---------------------- | -------------------------------------------------------- |
| `best_model.joblib`    | Trained prediction model                                 |
| `encoder.joblib`       | Converts model output back to class labels               |
| `feature_names.joblib` | Defines the expected input feature set and feature order |

---

## 📚 Documentation

Detailed documentation is available in the `docs/` directory.

| Document                                                 | Description                                                                      |
| -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 🏗️ [Project Architecture](docs/project_architecture.md) | Overview of the modular project structure and system design.                     |
| ⚙️ [MLOps Practices](docs/mlops_practices.md)            | MLOps workflow, experiment tracking, DVC, CI, and related engineering practices. |
| 📊 [Model Benchmark](docs/model_benchmark.md)            | Model comparison, evaluation metrics, and production model selection.            |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ryomen1234/Student-Academic-Outcome-Prediction.git

cd Student-Academic-Outcome-Prediction
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Training Pipeline

Run the project's training entry point:

```bash
python -m main
```

The project uses modular components for the machine learning workflow, including validation, preprocessing, training, evaluation, and artifact generation.

Because the source dataset was already clean, preprocessing and feature engineering remain intentionally limited.

---

## 🚀 Running the API Locally

The API application is located at:

```text
app/main.py
```

Start the API using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For development, you can use:

```bash
uvicorn app.main:app --reload
```

The API is then available at:

```text
http://localhost:8000
```

### Interactive Documentation

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

## 🎯 Using the API

The API exposes two main endpoints.

### Health Check

#### `GET /`

Check whether the API is running:

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

---

### Student Prediction

#### `POST /predict/file`

The endpoint accepts a CSV file and returns a prediction for every row.

The multipart form field must be named:

```text
file
```

Example:

```bash
curl -X POST \
  -F "file=@test_sample.csv" \
  http://localhost:8000/predict/file
```

### Input Requirements

The uploaded CSV must:

* Use `;` as the column delimiter.
* Contain the expected feature names.
* Not contain unexpected columns.
* Not omit required features.
* Preserve the expected data representation used during training.

The API reads the file using:

```python
pd.read_csv(file.file, sep=";")
```

The expected feature set and ordering are loaded from:

```text
artifacts/feature_names.joblib
```

The validator checks for missing and unexpected columns and reorders columns to match the expected training feature order.

### Example Response

```json
{
  "filename": "test_sample.csv",
  "rows": 10,
  "predictions": [
    "Dropout",
    "Graduate",
    "Enrolled"
  ]
}
```

The prediction order corresponds to the order of rows in the uploaded CSV.

### API Error Responses

#### `400 Bad Request`

Returned when input validation fails.

Example:

```json
{
  "detail": "Missing required columns: ['Course']"
}
```

Another possible validation response:

```json
{
  "detail": "Unexpected columns found: ['Unknown feature']"
}
```

#### `500 Internal Server Error`

Returned when an unexpected error occurs during prediction:

```json
{
  "detail": "Internal server error."
}
```

---

## 🐳 Running with Docker

The project includes a Dockerfile for packaging the FastAPI application together with its required dependencies, source code, configuration, and model artifacts.

The Docker image uses:

```dockerfile
FROM python:3.11-slim
```

The image also installs:

```text
libgomp1
```

which is required by native machine-learning components using the GNU OpenMP runtime.

The container starts the FastAPI application using:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 1. Build the Docker Image

Run this from the project root:

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

### 2. Run the Container

```bash
docker run --name student-success-api -p 8000:8000 student-success-api
```

The mapping:

```text
8000:8000
```

publishes host port `8000` to container port `8000`.

The API is then available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

### 3. Test the Container

Health check:

```bash
curl http://localhost:8000/
```

Prediction:

```bash
curl -X POST \
  -F "file=@test_sample.csv" \
  http://localhost:8000/predict/file
```

### 4. View Container Logs

```bash
docker logs student-success-api
```

Follow logs continuously:

```bash
docker logs -f student-success-api
```

### 5. Rebuild After Code Changes

If application code or dependencies change:

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

### Docker Troubleshooting

#### Container name already in use

Check existing containers:

```bash
docker ps -a
```

Remove the existing container:

```bash
docker rm -f student-success-api
```

#### `Attribute "app" not found`

The FastAPI application is located at:

```text
app/main.py
```

The correct Uvicorn module path is:

```text
app.main:app
```

Do not use:

```text
main:app
```

unless the FastAPI application object is actually defined in the root `main.py`.

#### `libgomp.so.1` not found

Rebuild the image so that the `libgomp1` system package is installed:

```bash
docker build --no-cache -t student-success-api .
```

#### API is not accessible

Check whether the container is running:

```bash
docker ps
```

Check the logs:

```bash
docker logs student-success-api
```

Verify that the port mapping contains:

```text
0.0.0.0:8000->8000/tcp
```

Then open:

```text
http://localhost:8000/
```

---

## 🔮 Future Improvements

The current implementation focuses on the machine learning workflow, MLOps components, local API inference, and Docker-based containerization.

Planned improvements include:

* ☁️ Cloud-based deployment on AWS / Azure / GCP
* 🔄 Automated deployment / CD pipeline
* 📦 Remote DVC storage for datasets and artifacts
* 📊 Model monitoring and performance tracking
* ♻️ Automated model retraining

These are planned extensions and are **not part of the current implementation**.

---

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. It helps others discover the project and supports future development.

```
```
