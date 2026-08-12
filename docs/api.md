# Student Success Prediction API

## Overview

The Student Success Prediction API is a FastAPI service that accepts student data as a CSV file and returns a predicted student outcome for each row.

The prediction classes configured for the project are:

- `Dropout`
- `Enrolled`
- `Graduate`

The API is designed around a small inference pipeline:

```text
CSV Upload
    |
    v
Read CSV (; delimiter)
    |
    v
Data Cleaning
    |
    v
Input Validation
    |
    v
Model Prediction
    |
    v
Label Decoding
    |
    v
JSON Response
```

The FastAPI application initializes the model service and validator during application startup. The application exposes a health endpoint and a file-based prediction endpoint.

---

## Technology Stack

| Component | Technology |
|---|---|
| API framework | FastAPI 0.116.1 |
| ASGI server | Uvicorn 0.35.0 |
| Data processing | pandas 2.3.1 |
| Numerical computing | NumPy 2.3.1 |
| ML utilities | scikit-learn 1.7.0 |
| Model serialization | joblib 1.5.1 |
| Model | LightGBM (`v1`) |
| Experiment tracking | Weights & Biases |
| Configuration | YAML / PyYAML |
| Validation | Custom `ValidateStudentData` |
| Python | 3.11 in Docker |

---

# API

## Base URL

### Local

```text
http://localhost:8000
```

When running the Docker container with port `8000` published:

```text
http://localhost:8000
```

---

# 1. Health Check

## `GET /`

Checks whether the API application is running.

### Request

```bash
curl http://localhost:8000/
```

### Response

```json
{
  "status": "healthy",
  "message": "Application is working fine."
}
```

### Status Code

| Code | Meaning |
|---|---|
| `200` | API is running |

---

# 2. Student Prediction

## `POST /predict/file`

Uploads a CSV file containing student data and returns a prediction for every row.

### Request

The endpoint expects a multipart form upload with the field name:

```text
file
```

Example:

```bash
curl -X POST \
  -F "file=@test_sample.csv" \
  http://localhost:8000/predict/file
```

### Python Example

```python
import requests

with open("test_sample.csv", "rb") as file:
    response = requests.post(
        "http://localhost:8000/predict/file",
        files={"file": file},
    )

print(response.json())
```

---

## Input Format

The API reads the uploaded CSV using:

```python
pd.read_csv(file.file, sep=";")
```

Therefore, the input file must use a **semicolon (`;`) as the column delimiter**.

The supplied test sample contains 10 student records.

The sample input contains fields including:

- `Marital status`
- `Application mode`
- `Application order`
- `Course`
- `Previous qualification`
- `Previous qualification (grade)`
- `Mother's qualification`
- `Father's qualification`
- `Mother's occupation`
- `Father's occupation`
- `Admission grade`
- `Debtor`
- `Tuition fees up to date`
- `Gender`
- `Scholarship holder`
- `Age at enrollment`
- `Curricular units 1st sem (enrolled)`
- `Curricular units 1st sem (evaluations)`
- `Curricular units 1st sem (approved)`
- `Curricular units 1st sem (grade)`
- `Inflation rate`

The final accepted feature set is determined by the stored `feature_names.joblib` artifact. The validator compares uploaded columns against this expected feature set before inference.

---

## Data Processing

The request goes through the following stages.

### 1. CSV loading

The uploaded file is converted to a pandas DataFrame using the semicolon delimiter.

### 2. Data cleaning

The DataFrame is passed to the project's `clean_data()` function together with the loaded configuration.

### 3. Feature validation

`ValidateStudentData` loads the expected feature names from:

```text
artifacts/feature_names.joblib
```

The validator checks:

- The input is a pandas DataFrame.
- No required feature is missing.
- No unexpected feature is present.
- Columns are reordered to match the training feature order.

### 4. Prediction

`Modelservice` loads:

```text
artifacts/models/best_model.joblib
artifacts/encoder.joblib
```

The model generates numeric predictions, and the encoder converts those predictions back to their original class labels.

---

# Response

A successful prediction returns JSON in the following format:

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

The exact prediction values depend on the model artifact currently stored in the project.

### Response Fields

| Field | Type | Description |
|---|---|---|
| `filename` | string | Name of the uploaded CSV file |
| `rows` | integer | Number of predictions generated |
| `predictions` | array of strings | Predicted class for each input row |

The prediction order corresponds to the order of rows in the uploaded CSV.

---

# Error Handling

## `400 Bad Request`

Returned when input validation fails.

Example:

```json
{
  "detail": "Missing required columns: ['Course']"
}
```

Another possible validation error is:

```json
{
  "detail": "Unexpected columns found: ['Unknown feature']"
}
```

Common causes:

- Missing required columns
- Unexpected columns
- Invalid input data

---

## `500 Internal Server Error`

Returned when an unexpected error occurs during prediction.

```json
{
  "detail": "Internal server error."
}
```

The application logs the underlying exception for debugging.

---

# Interactive API Documentation

FastAPI automatically provides interactive documentation.

After starting the API, open:

### Swagger UI

```text
http://localhost:8000/docs
```

### ReDoc

```text
http://localhost:8000/redoc
```

Swagger UI can be used to upload a CSV file and test `/predict/file` directly from the browser.

---

# Configuration

The project configuration contains the preprocessing, data paths, artifact paths, model configuration, and training metadata.

Important runtime artifact paths include:

```yaml
artifacts:
  model_path: artifacts/models/best_model.joblib
  encoder_path: artifacts/encoder.joblib
  feature_names_path: artifacts/feature_names.joblib
```

The configured model is:

```yaml
model:
  name: LightGBM
  version: v1
```

The configured prediction classes are:

```yaml
target:
  - Dropout
  - Enrolled
  - Graduate
```

---

# Model Artifacts

The inference service depends on three main artifacts.

| Artifact | Purpose |
|---|---|
| `best_model.joblib` | Trained prediction model |
| `encoder.joblib` | Converts model output back to class labels |
| `feature_names.joblib` | Defines the expected input feature set and feature order |

The artifact loader caches these resources so they are loaded once and reused by subsequent requests.

---

# Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For development, you can use:

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://localhost:8000/docs
```

---

# API Testing

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

---

# Important Input Requirement

The prediction endpoint is **not a generic CSV endpoint**. The uploaded CSV must contain exactly the feature set expected by the stored model artifacts after the project's cleaning step.

In particular:

- Use `;` as the CSV delimiter.
- Use the expected feature names.
- Do not add arbitrary columns.
- Do not omit required features.
- Preserve the expected data representation used during training.
