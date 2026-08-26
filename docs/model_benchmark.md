# Model Benchmark Results

## 1. Overview

Before selecting the final model, six classification algorithms were evaluated using the training dataset.

Each candidate model was evaluated using **Stratified K-Fold Cross-Validation**. The primary model-selection metric was **mean Macro F1**, with **Balanced Accuracy** used as an additional evaluation metric.

The benchmark evaluated:

* Logistic Regression
* Random Forest
* HistGradientBoosting
* XGBoost
* LightGBM
* CatBoost

The objective was to identify the model that provided the strongest overall performance across the three target classes.

---

## 2. Benchmark Results

| Model                | Macro F1 Mean | Macro F1 Std | Balanced Accuracy Mean | Balanced Accuracy Std |
| -------------------- | ------------: | -----------: | ---------------------: | --------------------: |
| **Random Forest**    |    **0.7165** |       0.0232 |                 0.7113 |                0.0231 |
| HistGradientBoosting |        0.7110 |       0.0168 |                 0.7031 |                0.0168 |
| Logistic Regression  |        0.7095 |       0.0145 |             **0.7169** |                0.0155 |
| XGBoost              |        0.7070 |       0.0202 |                 0.6990 |                0.0185 |
| LightGBM             |        0.7068 |       0.0140 |                 0.6988 |                0.0141 |
| CatBoost             |        0.6967 |       0.0122 |                 0.6881 |                0.0104 |

---

## 3. Model Selection

### 3.1 Macro F1

Random Forest achieved the highest mean Macro F1:

**Macro F1 = 0.7165**

The second-best model was HistGradientBoosting with a Macro F1 of **0.7110**.

The difference between the two models was approximately **0.0055**.

Since Macro F1 was selected as the primary model-selection metric, **Random Forest was selected as the best candidate model**.

---

### 3.2 Balanced Accuracy

Logistic Regression achieved the highest Balanced Accuracy:

**Balanced Accuracy = 0.7169**

Random Forest achieved:

**Balanced Accuracy = 0.7113**

Therefore, Logistic Regression performed slightly better than Random Forest according to Balanced Accuracy.

However, Random Forest achieved the highest **Macro F1**, which was the primary selection criterion.

This resulted in the following decision:

> **Random Forest was selected as the candidate model because it achieved the highest mean Macro F1 across the evaluated models.**

---

## 4. Stability Across Cross-Validation

The standard deviation of the cross-validation scores provides an indication of how much model performance varied between folds.

For Random Forest:

* Macro F1 standard deviation: **0.0232**
* Balanced Accuracy standard deviation: **0.0231**

The results show that Random Forest maintained relatively consistent performance across the validation folds.

The benchmark therefore provided both:

1. An estimate of average model performance.
2. An indication of performance variation across folds.

---

## 5. Model Ranking

Based on mean Macro F1, the models were ranked as follows:

| Rank | Model                |   Macro F1 |
| ---: | -------------------- | ---------: |
|    1 | **Random Forest**    | **0.7165** |
|    2 | HistGradientBoosting |     0.7110 |
|    3 | Logistic Regression  |     0.7095 |
|    4 | XGBoost              |     0.7070 |
|    5 | LightGBM             |     0.7068 |
|    6 | CatBoost             |     0.6967 |

The benchmark shows that the difference between several candidate models was relatively small. However, Random Forest achieved the highest Macro F1 and was therefore selected for the next stage of the training process.

---

## 6. Benchmark Decision

The benchmark stage resulted in the following decision:

```text
Candidate Models
       |
       v
Stratified Cross-Validation
       |
       v
Compare Macro F1
       |
       v
Random Forest
Macro F1 = 0.7165
       |
       v
Selected for Further Optimization
```

Random Forest was subsequently passed to the hyperparameter-tuning stage before final training.
