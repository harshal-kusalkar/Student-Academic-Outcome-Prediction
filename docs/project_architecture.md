Project Architecture
Overview

The Student Academic Dropout, Enrollment & Graduation Prediction project is designed using a modular and scalable architecture that separates each stage of the machine learning lifecycle into independent components. Rather than implementing the entire workflow in a single script, every module is responsible for a specific task such as data preprocessing, model training, evaluation, or deployment.

This separation of concerns improves maintainability, simplifies testing, and allows individual components to evolve independently without affecting the rest of the system. The architecture also follows configuration-driven development, enabling experiments and pipeline behavior to be modified through YAML configuration files instead of changing source code.

Architecture Goals

The project architecture is designed to achieve the following objectives:

Modular and reusable codebase
Clear separation of responsibilities
Configuration-driven workflows
Reproducible machine learning experiments
Easy integration of new models and preprocessing techniques
Production-ready deployment support
Maintainable and scalable project structure

High-Level Architecture


The architecture follows a sequential machine learning workflow where each component receives the output of the previous stage. This design enables the pipeline to be executed, tested, and extended with minimal coupling between modules.

Project Directory Structure

Student-Academic-Dropout-Enrollment-Graduation-Prediction/
│
├── .github/                  # GitHub Actions workflows
├── .dvc/                     # DVC configuration
│
├── app/                      # FastAPI and Streamlit applications
├── artifacts/                # Trained models and generated artifacts
├── config/                   # YAML configuration files
├── data/                     # Raw and processed datasets
├── docs/                     # Project documentation
├── entity/                   # Configuration and entity definitions
├── notebooks/                # Research and experimentation
│
├── src/
│   ├── data/
│   ├── features/
│   ├── pipeline/
│   ├── models/
│   ├── evaluation/
│   
│
├── tests/                    # Unit tests
├── utils/                    # Helper utilities
│
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md

The project structure separates source code, configuration, experiments, deployment, documentation, and testing into dedicated directories, making the repository easier to navigate and maintain.

Architecture Components
Configuration Layer

The project follows a configuration-first approach.

All configurable parameters—including preprocessing options, model settings, file paths, and pipeline behavior—are stored in YAML configuration files.

This approach provides several benefits:

No hardcoded parameters
Easier experimentation
Better reproducibility
Simplified maintenance
Cleaner source code


Data Layer

The data layer is responsible for managing datasets used throughout the machine learning pipeline.

Its responsibilities include:

Managing raw datasets
Storing processed datasets
Maintaining dataset consistency
Supporting reproducible data versioning

The project uses DVC (Data Version Control) to version datasets and machine learning artifacts, enabling reproducible workflows across different environments.

Processing Layer

The processing layer prepares raw student records for model training.

Typical responsibilities include:

Data validation
Data cleaning
Feature preprocessing
Handling categorical variables
Preparing datasets for training

Each preprocessing step is isolated from model training, allowing preprocessing pipelines to evolve independently.

Model Layer

The model layer contains the machine learning algorithms and training logic.

Its responsibilities include:

Training multiple machine learning models
Hyperparameter optimization
Model comparison
Model serialization
Saving the best-performing model

The architecture supports evaluating multiple algorithms before selecting the final production model.

Evaluation Layer

The evaluation layer measures model performance using multiple classification metrics.

Responsibilities include:

Model validation
Performance reporting
Metric calculation
Model comparison
Best model selection

Evaluation is performed independently of training, making it easy to benchmark different algorithms consistently.

Deployment Layer

The deployment layer exposes the trained model for inference.

The project includes:

FastAPI REST API

These interfaces allow predictions to be served programmatically.

Supporting Infrastructure

Beyond the core machine learning pipeline, the project incorporates several infrastructure components that improve reliability and maintainability.

| Component      | Purpose                                |
| -------------- | -------------------------------------- |
| Git            | Source code version control            |
| GitHub         | Repository hosting and collaboration   |
| GitHub Actions | Continuous Integration (CI)            |
| DVC            | Dataset and artifact versioning        |
| Pytest         | Automated unit testing                 |
| Docker         | Environment consistency and deployment |
| YAML           | Centralized project configuration      |
| Python Logging | Monitoring and debugging               |


These tools help create a reproducible and production-oriented machine learning workflow.

Architectural Principles

The project follows several software engineering principles:

Separation of Concerns

Each module is responsible for a single part of the machine learning workflow, reducing dependencies between components.

Modularity

Individual components can be developed, tested, or replaced without affecting the rest of the system.

Scalability

New preprocessing techniques, models, or deployment methods can be integrated with minimal architectural changes.

Reproducibility

Configuration files, DVC pipelines, and experiment tracking ensure that training runs can be reproduced consistently.

Maintainability

A structured directory layout and isolated modules simplify debugging, collaboration, and long-term maintenance.