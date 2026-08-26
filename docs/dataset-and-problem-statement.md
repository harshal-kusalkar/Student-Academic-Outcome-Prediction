# 1. Data and Problem Statement

## 1.1 Problem Overview

Academic dropout is a significant challenge in higher education. Identifying students who are at risk of dropping out at an early stage can allow educational institutions to provide appropriate support and intervention.

This project applies machine learning to predict the academic outcome of undergraduate students. The objective is to use information available about a student to predict their eventual academic status and identify students who may require additional support.

The problem is formulated as a **three-class classification problem** with the following target classes:

* **Dropout** — students who leave their studies without completing the program.
* **Enrolled** — students who remain enrolled without graduating within the considered period.
* **Graduate** — students who successfully complete their program.

The dataset has a strong class imbalance, with one of the three classes representing a substantially larger proportion of the observations.

---

## 1.2 Dataset Description

The dataset was created from information collected by a higher education institution in Portugal. The original dataset was constructed by combining information from several disjoint databases.

Each instance represents **one student** enrolled in an undergraduate degree program.

The students belong to different fields of study, including:

* Agronomy
* Design
* Education
* Nursing
* Journalism
* Management
* Social Service
* Technologies

The dataset contains information available throughout the student's academic journey, including:

* Academic background
* Demographic characteristics
* Socio-economic factors
* Academic performance during the first semester
* Academic performance during the second semester

These attributes provide information that can be used to learn patterns associated with different academic outcomes.

---

## 1.3 Dataset Purpose

The dataset was created as part of a project aimed at reducing academic dropout and failure in higher education.

The underlying objective is to use machine learning to identify students at risk at an early stage of their academic path. Early identification allows educational institutions to potentially introduce targeted support strategies before students ultimately drop out.

Therefore, the machine learning problem can be viewed as an **academic outcome prediction and early-risk identification problem**.

---

## 1.4 Machine Learning Problem

Given a set of attributes describing a student, the model predicts the student's final academic outcome.

The prediction task can be represented as:

```text
Student Data
     |
     +-----------------------+
     |                       |
     v                       v
Academic Information   Demographic / Socio-economic
     |                       |
     +-----------+-----------+
                 |
                 v
        Classification Model
                 |
        +--------+--------+
        |        |        |
        v        v        v
     Dropout  Enrolled  Graduate
```

The formal problem is:

> **Input:** A set of features describing a student.

> **Output:** One of three academic outcome classes: `Dropout`, `Enrolled`, or `Graduate`.

This makes the problem a **multiclass classification task**.

---

# 2. Data Characteristics

## 2.1 Dataset Size

As part of the project's data validation stage, the dataset was verified to contain:

| Property                 |    Value |
| ------------------------ | -------: |
| Rows                     |    4,424 |
| Columns                  |       37 |
| Duplicate rows           |        0 |
| Missing values           |        0 |
| Target column            | `Target` |
| Number of target classes |        3 |

The validation process completed successfully with the status:

```text
Data Validation: PASSED
```

The target variable contains the following three classes:

```text
Dropout
Enrolled
Graduate
```

---

## 2.2 Data Quality

Data validation is performed before the dataset enters the downstream machine learning pipeline.

The validation result confirms that the dataset used by the project contains:

* **4,424 student records**
* **37 columns**
* **No duplicate rows**
* **No missing values**
* A valid three-class target column

This validation step provides an explicit quality gate before the data is used for subsequent processing and model training.

---

# 3. Data Cleaning

After validation, the dataset passes through a dedicated data-cleaning stage.

The cleaning logic is implemented in `data_cleaning.py`.

The responsibility of this component is deliberately limited to preparing the raw dataset for downstream processing.

### Cleaning responsibilities

The data-cleaning stage performs the following operations:

1. **Normalize column names**
2. **Remove configured columns**
3. **Remove duplicate rows**
4. **Return the cleaned DataFrame**

The cleaning process can be represented as:

```mermaid
flowchart LR
    A[Raw Dataset] --> B[Normalize Column Names]
    B --> C[Remove Configured Columns]
    C --> D[Remove Duplicate Rows]
    D --> E[Clean DataFrame]
```

### Why separate data cleaning from validation?

Data validation and data cleaning serve different purposes.

**Validation** determines whether the data satisfies the expected quality requirements.

**Cleaning** transforms the raw data into a consistent format that can be consumed by subsequent stages of the pipeline.

This separation keeps the pipeline modular and makes it easier to identify whether a problem originates from the input data or from a transformation step.

---

# 4. Data Split

The dataset is divided into two subsets:

| Dataset  | Percentage |
| -------- | ---------: |
| Training |        80% |
| Test     |        20% |

The **training dataset** is used to train the machine learning model.

The **test dataset** is kept separate and is used to evaluate the final model on previously unseen data.

This separation is important because evaluating the model on the same data used during training would not provide a reliable estimate of its generalization performance.

---

# 5. Class Imbalance

The target variable contains three classes:

* `Dropout`
* `Enrolled`
* `Graduate`

The dataset has a **strong class imbalance**, meaning that the number of observations is not evenly distributed across these classes.

This is an important consideration when developing and evaluating the model.

A model trained on imbalanced data may favor the majority class and achieve a relatively high overall accuracy while performing poorly on minority classes.

Therefore, model evaluation should not rely solely on accuracy. Class-level metrics such as precision, recall, F1-score, and the confusion matrix should also be considered.

---

# 6. Project Objective

The objective of the machine learning system is:

> **To predict the academic outcome of a student as Dropout, Enrolled, or Graduate, with the broader goal of enabling early identification of students who may be at risk of dropping out.**

The resulting prediction can support educational institutions in identifying students who may benefit from additional academic or institutional support.

The complete data-to-model process therefore begins with **validated and cleaned student data**, which is subsequently passed to the training pipeline for feature processing, model training, evaluation, and model management.
