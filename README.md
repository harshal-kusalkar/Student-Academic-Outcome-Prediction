# 🎓 Student Academic Outcome Prediction

A modular MLOps project that predicts whether a student is likely to **Dropout**, **Remain Enrolled**, or **Graduate** using academic, demographic, and socioeconomic features.

> **Implementation Scope:** This repository focuses on MLOps component architecture, local API inference, and Docker containerization. Minimal preprocessing was applied since the raw dataset was already clean. Automated cloud deployment (CD) and cloud hosting are planned as future work.

---

## 🛠️ Tech Stack

- **ML Frameworks:** Scikit-learn, LightGBM, XGBoost, CatBoost
- **MLOps & Pipeline:** DVC (Versioning), Weights & Biases (Tracking), Optuna (HPO)
- **API & Containerization:** FastAPI, Uvicorn, Docker
- **CI & Testing:** GitHub Actions, Pytest

---

## 📊 Model Performance & Benchmarks

Models were evaluated using **Macro F1-Score** due to class imbalance[cite: 1]. **LightGBM** is selected as the configured runtime model[cite: 1].

| Rank | Model | Accuracy | Macro F1 | Status |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **XGBoost** | **0.7409** | **0.6642** | Finalist[cite: 1] |
| 🥈 | **LightGBM** | **0.7409** | **0.6605** | **Production Model**[cite: 1] |
| 🥉 | **CatBoost** | **0.7369** | **0.6520** | Runner-up[cite: 1] |

**Inference Artifacts:** `best_model.joblib`, `encoder.joblib`, and `feature_names.joblib` located under `artifacts/`[cite: 1].

---

## 📁 Project Structure

```text
Student-Academic-Outcome-Prediction/
├── app/                  # FastAPI entry point (main.py)
├── artifacts/            # Model, encoder, and feature schemas
├── config/               # Project configuration files
├── src/                  # Source modules (data, inference)
├── tests/                # Pytest unit and integration tests
├── Dockerfile            # Container configuration
├── dvc.yaml              # DVC pipeline stages
└── main.py               # Local pipeline entry point

🚀 Quickstart
1. Installation