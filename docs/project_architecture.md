# Project Architecture

## Overview

The project follows a **modular, configuration-driven architecture** that separates each stage of the machine learning lifecycle into independent components. This design improves maintainability, simplifies testing, and makes it easier to extend the pipeline with new models, preprocessing techniques.

---

## Architecture Goals

The architecture is designed around the following principles:

- Modular and reusable components
- Clear separation of responsibilities
- Configuration-driven workflows
- Reproducible experiments
- Scalable project structure

---

## High-Level Architecture

```mermaid
flowchart LR

    A[Dataset]
    B[Data Validation]
    C[Data Preprocessing]
    D[Model Training]
    E[Model Evaluation]
    F[Model Artifacts]
    G[FastAPI]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

Each stage performs a single responsibility and passes its output to the next stage, resulting in a modular and reproducible machine learning pipeline.

---

## Project Structure

```text
Student-Academic-Dropout-Enrollment-Graduation-Prediction/
│
├── .github/              # CI workflows
├── .dvc/                 # DVC configuration
├── app/                  # FastAPI application
├── artifacts/            # Models and generated artifacts
├── config/               # YAML configuration
├── data/                 # Raw and processed datasets
├── docs/                 # Documentation
├── entity/               # Configuration entities
├── notebooks/            # Research & experimentation
├── src/
│   ├── data/
│   ├── pipeline/
│   ├── models/
│   
├── tests/                # Unit tests
├── utils/                # Utility modules
├── Dockerfile
├── dvc.yaml
├── main.py
└── README.md
```

The project structure separates source code, configuration, datasets, artifacts, documentation, and testing into dedicated modules, making the repository easier to maintain and extend.

---

# Core Components

## Configuration

Pipeline behavior is managed through **YAML configuration files** instead of hardcoded values.

**Responsibilities**

- Dataset paths
- Training configuration
- Hyperparameters
- Model settings
- Pipeline configuration

**Benefits**

- Easier experimentation
- Improved reproducibility
- Cleaner codebase

---

## Data Processing

Responsible for preparing data before model training.

**Responsibilities**

- Data validation
- Data preprocessing
- Feature engineering
- Dataset preparation

DVC is used to version datasets and generated artifacts, ensuring reproducible experiments.

---

## Model Development

Implements the complete model development workflow.

**Responsibilities**

- Train multiple models
- Hyperparameter optimization
- Model comparison
- Save production model

The modular design allows new algorithms to be added without changing the overall pipeline.

---

## Model Evaluation

Evaluates candidate models using standardized metrics.

**Responsibilities**

- Performance evaluation
- Metric calculation
- Model comparison
- Production model selection

Separating evaluation from training ensures a fair and consistent benchmarking process.

---

## Deployment

Provides production inference through:

- FastAPI REST API

The deployment layer consumes the trained model artifacts and exposes prediction endpoints for inference.

---

# Supporting Infrastructure

| Component | Purpose |
|-----------|---------|
| Git | Source code version control |
| GitHub | Repository hosting |
| GitHub Actions | Continuous Integration |
| DVC | Data & artifact versioning |
| Pytest | Automated testing |
| Docker | Containerization |
| YAML | Configuration management |
| Python Logging | Application logging |

Together, these tools provide a reproducible and production-oriented development workflow.

---

# Architectural Principles

| Principle | Description |
|-----------|-------------|
| **Modularity** | Independent components with single responsibilities |
| **Separation of Concerns** | Data, training, evaluation, and deployment are isolated |
| **Reproducibility** | Configuration files, DVC, and experiment tracking ensure consistent experiments |
| **Scalability** | New models and preprocessing steps can be integrated with minimal changes |
| **Maintainability** | Organized project structure simplifies debugging and collaboration |

---

> [!NOTE]
> The architecture is designed to support the complete machine learning lifecycle while remaining modular, reproducible, and easy to maintain.