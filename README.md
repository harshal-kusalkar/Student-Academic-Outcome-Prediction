# Student Academic Outcome Prediction

A modular machine learning project for **early prediction of student
academic outcomes** using academic, demographic, and socioeconomic data.
The goal is to help educational institutions identify students who may
require timely academic or financial support.

The project predicts three outcomes:

- **Dropout**
- **Enrolled**
- **Graduate**

## Highlights

- Multi-class student outcome prediction
- Configuration-driven data validation and preprocessing
- Benchmarking of **6 machine learning models**
- **XGBoost** as the current model
- Hyperparameter optimization with **Optuna** in notebooks
- Experiment tracking with **Weights & Biases (W&B)** in notebooks
- Data, pipeline, and artifact versioning with **DVC**
- FastAPI inference service with CSV upload
- Dockerized **inference-only** deployment
- GitHub Actions for **continuous integration (CI)**
- Unit tests for data validation, cleaning, and splitting

## Dataset

The dataset contains information available at student enrollment,
including academic, demographic, and socioeconomic factors. Each row
represents a student.

| Class      | Meaning                       |
|------------|-------------------------------|
| `Dropout`  | Student leaves the program    |
| `Enrolled` | Student remains enrolled      |
| `Graduate` | Student completes the program |

### Data Preparation

- Data validation checks structure, missing values, duplicates, and
  target labels.
- Column names are normalized.
- Configured features are removed during cleaning.
- Second-semester outcome-related features are excluded to avoid using
  information unavailable at the intended prediction point.
- Data is split into **80% training** and **20% testing** with
  `random_state=42`.
- The dataset contains no missing values.

## Model Development

Six classifiers were benchmarked using the same evaluation setup:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost

**Macro F1** is the primary model-selection metric because the target
classes are imbalanced.

### Benchmark Results

| Model               |   Accuracy | Balanced Accuracy |   Macro F1 |
|---------------------|-----------:|------------------:|-----------:|
| **XGBoost**         | **0.7409** |        **0.6567** | **0.6642** |
| LightGBM            |     0.7409 |            0.6540 |     0.6605 |
| CatBoost            |     0.7369 |            0.6458 |     0.6520 |
| Random Forest       |     0.7420 |            0.6432 |     0.6502 |
| Extra Trees         |     0.7279 |            0.6250 |     0.6323 |
| Logistic Regression |     0.6878 |            0.5673 |     0.5567 |

XGBoost achieved the strongest **Macro F1** and balanced accuracy in the
benchmark and is the current model used by the main pipeline.

### Hyperparameter Optimization

Optuna optimization was performed separately in notebooks. The resulting
parameters were retained for use by the main training pipeline.

| Parameter          |       Value |
|--------------------|------------:|
| Best trial         |        `75` |
| Best CV Macro F1   |   `0.68070` |
| `n_estimators`     |       `793` |
| `learning_rate`    | `0.0758589` |
| `max_depth`        |         `4` |
| `min_child_weight` |         `2` |
| `subsample`        |  `0.897915` |
| `colsample_bytree` |  `0.744593` |
| `gamma`            |  `0.723508` |
| `reg_alpha`        |  `0.003037` |
| `reg_lambda`       | `0.0000159` |

> **Note:** `0.68070` is the best cross-validation score from Optuna,
> not the held-out test-set score.

## Experimentation vs. Main Pipeline

Notebook experimentation is intentionally separated from the main
pipeline.

**Notebooks** are used for:

- Model benchmarking
- Optuna hyperparameter optimization
- W&B experiment tracking

The selected XGBoost parameters are then reused by the **main training
pipeline**, which produces the model artifact used for inference.

## MLOps

| Practice                    | Implementation  |
|-----------------------------|-----------------|
| Configuration               | YAML + Pydantic |
| Data & pipeline versioning  | DVC             |
| Experiment tracking         | W&B             |
| Hyperparameter optimization | Optuna          |
| Testing                     | Pytest          |
| Continuous Integration      | GitHub Actions  |
| API                         | FastAPI         |
| Containerization            | Docker          |

The project currently focuses on reproducible development, training,
evaluation, and inference. **Model monitoring, Kubernetes deployment,
and automated retraining are not implemented.**

## DVC Pipeline

The DVC pipeline contains three stages:

1.  **Data** — prepares processed train/test datasets.
2.  **Train** — trains the configured XGBoost model and stores model
    artifacts.
3.  **Evaluate** — evaluates the trained model and stores evaluation
    reports.

Run the pipeline with:

``` bash
dvc repro
```

Key generated artifacts include:

``` text
artifacts/
├── models/
│   └── best_model.joblib
├── feature_names.joblib
├── eval_result.json
└── classification_report.joblib
```

## Project Structure

``` text
Student-Academic-Outcome-Prediction/
├── .github/              # GitHub Actions CI
├── .dvc/                 # DVC metadata
├── app/                  # FastAPI application
├── artifacts/            # Models and generated artifacts
├── config/               # YAML configuration
├── data/                 # Raw and processed data
├── docs/                 # Detailed documentation
├── entity/               # Configuration entities
├── notebooks/            # Model experimentation
├── src/                  # Data, training, evaluation, inference
├── tests/                # Automated tests
├── utils/                # Shared utilities
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/harshal-kusalkar/Student-Academic-Outcome-Prediction
cd Student-Academic-Outcome-Prediction
```

### 2. Create a virtual environment

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

## Training

Run the main training pipeline:

``` bash
python -m main
```

Or reproduce the DVC pipeline:

``` bash
dvc repro
```

The optimized XGBoost parameters obtained during notebook
experimentation are reused by the main pipeline.

## FastAPI Inference

Start the API locally:

``` bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For development:

``` bash
uvicorn app.main:app --reload
```

Open the interactive documentation at:

``` text
http://localhost:8000/docs
```

### Health Check

``` bash
curl http://localhost:8000/
```

### Prediction

The prediction endpoint accepts a CSV file using a **semicolon (`;`)
delimiter**:

``` bash
curl -X POST \
  -F "file=@test_sample.csv" \
  http://localhost:8000/predict/file
```

A successful response contains:

``` json
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

The API validates the uploaded feature set against the stored feature
names before prediction and rejects missing or unexpected features.

## Docker — Inference Only

Docker packages the **FastAPI inference service**, runtime dependencies,
configuration, source modules, and model artifacts.

Build the image:

``` bash
docker build -t student-success-api .
```

Run the container:

``` bash
docker run --name student-success-api -p 8000:8000 student-success-api
```

The API is available at:

``` text
http://localhost:8000
```

Swagger UI:

``` text
http://localhost:8000/docs
```

The container uses Python 3.11 and runs the application as a non-root
user.

## Testing

Testing currently focuses on the data-processing layer.

Implemented tests cover:

- Data validation
- Missing-value and duplicate detection
- Target-label validation
- Data cleaning
- Configured column removal
- Column-name normalization
- Train/test splitting
- Split reproducibility

Run tests with:

``` bash
pytest tests/ -v
```

Mock-based tests and broader model/API integration tests are **not
currently implemented**.

## Continuous Integration

GitHub Actions is used for **CI only**.

The workflow runs automated tests for pushes and pull requests targeting
the `main` branch. There is currently no automated deployment workflow.

## Documentation

Additional documentation is available in [`docs/`](docs/):

- [`API Documentation`](docs/api.md)
- [`Docker Documentation`](docs/docker.md)
- [`MLOps Practices`](docs/mlops_practices.md)
- [`Model Benchmark`](docs/model_benchmark.md)
- [`Project Architecture`](docs/project_architecture.md)

> Some documents contain earlier implementation details. The main README
> reflects the current XGBoost-based implementation.

## Limitations

- Prediction quality depends on the available enrollment-time features
  and dataset.
- Testing currently covers selected data-processing components rather
  than the complete application.
- Notebook experimentation is separate from the main automated pipeline.
- Model monitoring is not implemented.
- Kubernetes deployment is not implemented.
- Automated model retraining is not implemented.

## Future Improvements

- Expand API and model integration tests
- Automate model retraining
- Add cloud deployment
- Improve model and artifact lifecycle management

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE)
for details.
