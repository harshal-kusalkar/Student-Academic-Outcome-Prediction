Model Benchmark
Overview

Model benchmarking is the process of evaluating multiple machine learning algorithms under the same experimental conditions to identify the most suitable model for production deployment.

In this project, six classification algorithms are trained using the same dataset, preprocessing pipeline, and evaluation strategy. Their performance is compared using multiple evaluation metrics to ensure an objective and fair selection process.

Objectives

The benchmarking process aims to:

Evaluate multiple machine learning algorithms
Compare model performance using standardized metrics
Identify the most suitable production model
Analyze the trade-off between predictive performance and computational efficiency
Support objective model selection

Models Evaluated

The following supervised machine learning algorithms were benchmarked during experimentation.

| Model               | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| Logistic Regression | Linear baseline classifier for multi-class classification                   |
| Random Forest       | Ensemble learning using multiple decision trees                             |
| Extra Trees         | Randomized tree-based ensemble method                                       |
| XGBoost             | Gradient boosting algorithm optimized for predictive performance            |
| LightGBM            | Histogram-based gradient boosting framework                                 |
| CatBoost            | Gradient boosting algorithm with efficient handling of categorical features |

All models were evaluated using identical preprocessing and evaluation procedures to ensure a fair comparison.

Evaluation Metrics

Model performance is assessed using multiple classification metrics.

| Metric            | Purpose                                                                               |
| ----------------- | ------------------------------------------------------------------------------------- |
| Accuracy          | Measures overall prediction correctness                                               |
| Balanced Accuracy | Accounts for class imbalance by averaging recall across classes                       |
| Macro F1-Score    | Evaluates overall classification performance by giving equal importance to each class |


Among these metrics, Macro F1-Score is used as the primary criterion for model selection because the dataset contains three target classes with an imbalanced distribution.

Benchmark Results

The following benchmark summarizes the performance of each evaluated model.

| Rank | Model               |  Accuracy  | Balanced Accuracy |  Macro F1  | Training Time (s) |
| :--: | ------------------- | :--------: | :---------------: | :--------: | ----------------: |
|  🥇  | **XGBoost**         | **0.7409** |     **0.6567**    | **0.6642** |              1.06 |
|  🥈  | **LightGBM**        |   0.7409   |       0.6540      |   0.6605   |              7.96 |
|  🥉  | **CatBoost**        |   0.7369   |       0.6458      |   0.6520   |             19.55 |
|   4  | Random Forest       |   0.7420   |       0.6432      |   0.6502   |          **0.87** |
|   5  | Extra Trees         |   0.7279   |       0.6250      |   0.6323   |          **0.80** |
|   6  | Logistic Regression |   0.6878   |       0.5673      |   0.5567   |              1.27 |

These benchmark results provide an objective comparison of predictive performance across all candidate models.

Hyperparameter Optimization

After benchmarking, the two highest-performing models were selected for hyperparameter optimization

| Model        | Best Macro F1 (CV) | Best Trial | Status     |
| ------------ | :----------------: | :--------: | ---------- |
| **XGBoost**  |     **0.6776**     |     19     | Finalist   |
| **LightGBM** |       0.6740       |     24     | ✅ Selected |

This optimization stage refines the performance of the leading candidates before making the final production decision.

Final Model Selection

Although XGBoost achieved the highest Macro F1-Score after hyperparameter optimization, LightGBM was selected as the production model.

The selection considered not only predictive performance but also deployment efficiency.

Selection Criteria
Competitive predictive performance
Faster inference
Lower memory consumption
Computational efficiency
Better suitability for production deployment

This balanced approach ensures the final model performs well while remaining efficient in real-world applications.

Benchmark Summary

| Category                    | Summary                               |
| --------------------------- | ------------------------------------- |
| Models Evaluated            | 6                                     |
| Primary Evaluation Metric   | Macro F1-Score                        |
| Hyperparameter Optimization | Optuna                                |
| Final Production Model      | LightGBM                              |
| Model Selection Strategy    | Performance and deployment efficiency |

Key Takeaways
Six machine learning algorithms were benchmarked using a consistent evaluation pipeline.
Macro F1-Score served as the primary metric for comparing model performance.
XGBoost achieved the highest benchmark score after optimization.
LightGBM was selected for production because it offered the best balance between predictive performance and computational efficiency.
Hyperparameter optimization further improved the performance of the top-performing models before the final selection.