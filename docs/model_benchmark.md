# Model Benchmark

## Overview

To identify the most suitable production model, multiple machine learning algorithms were evaluated using the same dataset, preprocessing pipeline, and evaluation strategy. This ensured a fair comparison and enabled objective model selection based on consistent performance metrics.

---

## Benchmark Workflow

```mermaid
flowchart LR

    A[Processed Dataset]
    B[Train Models]
    C[Evaluate Performance]
    D[Compare Results]
    E[Hyperparameter Optimization]
    F[Select Production Model]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

---

## Models Evaluated

The following supervised learning algorithms were benchmarked.

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline classifier |
| Random Forest | Bagging-based decision tree ensemble |
| Extra Trees | Randomized tree ensemble |
| XGBoost | Gradient boosting framework |
| LightGBM | Histogram-based gradient boosting |
| CatBoost | Gradient boosting with categorical feature support |

All models were trained using the same preprocessing pipeline and evaluated under identical conditions.

---

## Evaluation Metrics

Model performance was evaluated using multiple classification metrics.

| Metric | Purpose |
|---------|---------|
| Accuracy | Overall prediction accuracy |
| Balanced Accuracy | Handles class imbalance by averaging recall |
| Macro F1-Score | Equal importance to all target classes |

> [!IMPORTANT]
> **Macro F1-Score** was used as the primary model selection metric because the dataset contains an imbalanced distribution of **Dropout**, **Enrolled**, and **Graduate** classes.

---

## Benchmark Results

| Rank | Model | Accuracy | Balanced Accuracy | Macro F1 | Training Time (s) |
|:---:|----------------------|:-------:|:-----------------:|:--------:|------------------:|
| 🥇 | **XGBoost** | **0.7409** | **0.6567** | **0.6642** | 1.06 |
| 🥈 | **LightGBM** | 0.7409 | 0.6540 | 0.6605 | 7.96 |
| 🥉 | **CatBoost** | 0.7369 | 0.6458 | 0.6520 | 19.55 |
| 4 | Random Forest | 0.7420 | 0.6432 | 0.6502 | **0.87** |
| 5 | Extra Trees | 0.7279 | 0.6250 | 0.6323 | **0.80** |
| 6 | Logistic Regression | 0.6878 | 0.5673 | 0.5567 | 1.27 |

---

## Hyperparameter Optimization

The two highest-performing models were selected for hyperparameter optimization using **Optuna**.

| Model | Best Macro F1 (CV) | Best Trial | Status |
|--------|:------------------:|:----------:|--------|
| **XGBoost** | **0.6776** | 19 | Finalist |
| **LightGBM** | 0.6740 | 24 | ✅ Selected |

---

## Production Model Selection

Although **XGBoost** achieved the highest Macro F1-Score after optimization, **LightGBM** was selected as the final production model.

The decision considered both predictive performance and deployment efficiency.

### Selection Criteria

- Competitive predictive performance
- Faster inference
- Lower memory consumption
- Better computational efficiency
- Suitable for production deployment

---

## Benchmark Summary

| Category | Result |
|-----------|--------|
| Models Evaluated | 6 |
| Primary Metric | Macro F1-Score |
| Hyperparameter Optimization | Optuna |
| Final Production Model | LightGBM |
| Selection Strategy | Performance + Deployment Efficiency |

---

## Key Takeaways

- Benchmarked **6** machine learning algorithms using a consistent evaluation pipeline.
- Used **Macro F1-Score** as the primary metric for model comparison.
- Applied **Optuna** to optimize the top-performing models.
- Selected **LightGBM** for production based on its balance of predictive performance and computational efficiency.

> [!NOTE]
> The benchmarking process follows a structured and reproducible workflow, ensuring that the production model is selected through objective evaluation rather than empirical assumptions.