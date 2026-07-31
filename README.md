Project Name

Student Academic Dropout, Enrollment & Graduation Prediction

Project Description

This project is a machine learning system designed to predict whether a student is likely to drop out, remain enrolled, or graduate based on academic, demographic, and socioeconomic factors. By identifying students who are at risk of dropping out at an early stage, educational institutions can take proactive measures to improve student success.

The project follows a complete machine learning workflow, including data preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, and model selection, while maintaining a modular and reproducible codebase.

Objective

The primary objective of this project is to assist educational institutions in identifying students who are at a high risk of dropping out so that timely interventions can be provided, such as:

Academic guidance and mentoring
Financial assistance or scholarships
Counseling and mental health support
Career guidance
Personalized learning support

Early identification enables institutions to improve student retention rates, increase graduation rates, and provide targeted support where it is needed most.

3. Key Highlights

This project demonstrates the development of a production-oriented machine learning system by integrating modern MLOps practices, software engineering principles, and model optimization techniques.

🚀 Production-Ready Machine Learning Pipeline
Modular and scalable project architecture following software engineering best practices.
Clean separation of data processing, training, evaluation, and deployment components.
Configuration-driven workflow for easy experimentation and maintenance.
⚙️ Complete MLOps Workflow
End-to-end machine learning pipeline from data ingestion to model deployment.
Reproducible experiments using DVC for data and artifact versioning.
Experiment tracking with MLflow and Weights & Biases (W&B).
Automated training pipeline with configurable parameters.
🤖 Advanced Model Development
Extensive feature engineering and preprocessing pipeline.
Hyperparameter optimization using Optuna.
Benchmarking multiple machine learning models to select the best-performing model.
Automated model evaluation using multiple classification metrics.
📦 Deployment Ready
REST API built with FastAPI for serving predictions.
Interactive prediction interface using Streamlit.
Easily deployable architecture suitable for cloud or containerized environments.
🔄 CI/CD & Software Engineering
Automated testing using GitHub Actions.
Unit testing with pytest to improve code reliability.
Comprehensive logging system for debugging and monitoring.
Configuration management using YAML files.
Production-ready project structure following industry best practices.
📊 Experiment Management
Track training runs, evaluation metrics, and hyperparameters.
Compare multiple experiments and model versions.
Store artifacts, trained models, and performance reports for reproducibility.

🛠️ Technologies & Tools

| Category                    | Technologies                              |
| --------------------------- | ----------------------------------------- |
| Programming Language        | Python                                    |
| Machine Learning            | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Experiment Tracking         | MLflow, Weights & Biases (W&B)            |
| Hyperparameter Optimization | Optuna                                    |
| Data Versioning             | DVC                                       |
| API Development             | FastAPI                                   |
| Testing                     | Pytest                                    |
| CI/CD                       | GitHub Actions                            |
| Version Control             | Git, GitHub                               |
| Configuration               | YAML                                      |
| Logging                     | Python Logging                            |
| Data Processing             | Pandas, NumPy                             |
| Visualization               | Matplotlib, Seaborn                       |

4. Project Architecture

Student-Academic-Dropout-Enrollment-Graduation-Prediction/
│
├── .github/                  # GitHub Actions CI/CD workflows
├── .dvc/                     # DVC configuration
│
├── app/                      # FastAPI and Streamlit applications
├── artifacts/                # Trained models, metrics, and generated artifacts
├── config/                   # Project configuration files (YAML)
├── data/                     # Raw and processed datasets
├── docs/                     # Project documentation and images
├── entity/                   # Configuration and entity classes
├── notebooks/                # EDA, experimentation, and model development
├── src/                      # Core machine learning pipeline
│   ├── data/
│   ├── features/
│   ├── pipeline/
│   ├── models/
│   ├── evaluation/
│   └── deployment/
│
├── tests/                    # Unit tests
├── utils/                    # Utility functions and helper modules
│
├── Dockerfile                # Docker configuration
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # DVC pipeline lock file
├── main.py                   # Project entry point
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── .gitignore
├── .dvcignore
└── README.md


5. Features

The project provides a complete end-to-end machine learning workflow for predicting student academic outcomes, from data preparation to deployment.

📊 Data Validation
Validate dataset structure and feature consistency.
Detect missing or invalid values before training.
Ensure data quality for reliable model performance.
🧹 Data Preprocessing
Clean and transform raw student records.
Handle missing values and categorical features.
Prepare data for model training through an automated preprocessing pipeline.
🤖 Model Training
Train multiple machine learning algorithms using a modular training pipeline.
Support configurable training parameters through YAML configuration files.
Generate reproducible training results.
📈 Model Evaluation
Evaluate trained models using multiple classification metrics.
Analyze model performance on unseen test data.
Generate evaluation reports for performance comparison.
🏆 Model Comparison
Benchmark multiple machine learning models.
Compare models using key evaluation metrics.
Automatically identify and save the best-performing model.
🎯 Hyperparameter Optimization
Optimize model performance using Optuna.
Automate hyperparameter search to improve prediction accuracy.
Record optimization results for future analysis.
📚 Experiment Tracking
Track experiments with MLflow and Weights & Biases (W&B).
Log parameters, metrics, trained models, and artifacts.
Compare multiple experiments to ensure reproducibility.
🌐 REST API
Deploy the trained model using FastAPI.
Expose prediction endpoints for seamless integration with external applications.
Provide fast and scalable inference services.
🖥️ Interactive Web Application
User-friendly prediction interface built with Streamlit.
Allow users to input student information and receive predictions in real time.
Suitable for demonstrations and non-technical users.
🐳 Docker Support
Containerized application for consistent deployment across different environments.
Simplified setup and execution using Docker.
Enables reproducible development and production environments.
⚙️ Reproducible MLOps Pipeline
Version datasets and model artifacts using DVC.
Automate pipeline stages for reproducible machine learning workflows.
Integrate testing and continuous integration using GitHub Actions.

6. Tech Stack

This project leverages a modern machine learning and MLOps ecosystem to build, track, deploy, and maintain a production-ready prediction system.

| Category                        | Technologies           |
| ------------------------------- | ---------------------- |
| **Programming Language**        | Python                 |
| **Machine Learning**            | Scikit-learn           |
| **Data Processing**             | Pandas, NumPy          |
| **Hyperparameter Optimization** | Optuna                 |
| **Experiment Tracking**         | Weights & Biases (W&B) |
| **Model Deployment**            | FastAPI                |
| **Containerization**            | Docker                 |
| **MLOps & Data Versioning**     | DVC                    |
| **CI/CD**                       | GitHub Actions         |
| **Version Control**             | Git, GitHub            |
| **Configuration Management**    | YAML                   |
| **Testing**                     | Pytest                 |
| **Logging**                     | Python Logging         |

🔧 Technology Overview
Python – Primary programming language for the project.
Pandas & NumPy – Data cleaning, manipulation, and numerical computations.
Scikit-learn – Machine learning models, preprocessing, evaluation metrics, and pipelines.
Optuna – Automated hyperparameter optimization.
Weights & Biases (W&B) – Experiment tracking, metric visualization, and artifact management.
FastAPI – High-performance REST API for serving model predictions.
Docker – Containerization for consistent development and deployment.
DVC – Dataset, model, and pipeline version control.
GitHub Actions – Continuous Integration (CI) for automated testing and workflow execution.
Pytest – Unit testing to ensure code reliability.
YAML – Configuration-driven project management.
Python Logging – Centralized logging for debugging and monitoring.

7. Workflow / Pipeline

                     Student Dataset
                           │
                           ▼
                  Data Validation
          (Schema & Data Quality Checks)
                           │
                           ▼
                  Data Preprocessing
      (Cleaning, Encoding, Feature Selection)
                           │
                           ▼
                    Model Training
                           │
                           ▼
                   Model Evaluation
     (Accuracy, Precision, Recall, F1-Score)
                           │
                           ▼
              Experiment Tracking
               (Weights & Biases)
                           │
                           ▼
                 Save Best Model
          (Artifacts & DVC Versioning)
                           │
                           ▼
          FastAPI REST API / Streamlit App
                           │
                           ▼
                    Student Prediction

📓 Model Benchmarking & Experimentation

Model comparison is performed separately in the Jupyter notebooks during the experimentation phase. Multiple machine learning algorithms are trained and evaluated to benchmark their performance. The best-performing model is then selected for hyperparameter optimization and integrated into the production training pipeline.

This separation keeps the production pipeline lightweight while preserving a dedicated environment for research and experimentation.

BENCHMARK RESULT:
Ranking by Macro F1 (Primary Metric)
| Rank | Model              |   F1 Macro | Balanced Accuracy |   Accuracy |
| ---- | ------------------ | ---------: | ----------------: | ---------: |
| 🥇   | XGBoost            | **0.6642** |        **0.6567** |     0.7409 |
| 🥈   | LightGBM           | **0.6605** |        **0.6540** |     0.7409 |
| 🥉   | CatBoost           | **0.6520** |        **0.6458** |     0.7369 |
| 4    | RandomForest       |     0.6502 |            0.6432 | **0.7420** |
| 5    | ExtraTrees         |     0.6323 |            0.6250 |     0.7279 |
| 6    | LogisticRegression |     0.5567 |            0.5673 |     0.6878 |

AFTER EVALUATION RESULT AT WANDB: https://wandb.ai/h41497254-none/student-drop-enroll-grad-preds/runs/q69ddch9?nw=nwuserh41497254 

