# Final Model Evaluation Results

## 1. Overview

After model selection and final training, the resulting model was evaluated on the **untouched test dataset**.

The test dataset contains **885 student records** and was not used during model training or model selection.

The purpose of this evaluation is to measure how well the final trained model generalizes to previously unseen student data.

The evaluation includes:

* Accuracy
* Macro F1
* Balanced Accuracy
* Macro Precision
* Macro Recall
* Per-class precision, recall, and F1-score
* Confusion matrix

---

## 2. Overall Performance

| Metric            |      Score |
| ----------------- | ---------: |
| Accuracy          | **0.7537** |
| Macro F1          | **0.7077** |
| Balanced Accuracy | **0.7121** |
| Macro Precision   | **0.7104** |
| Macro Recall      | **0.7121** |

The final model achieved an overall accuracy of approximately **75.37%** on the test dataset.

The **Macro F1 score of 0.7077** indicates the model's average F1 performance across the three classes.

Balanced Accuracy was **0.7121**, while Macro Precision and Macro Recall were **0.7104** and **0.7121**, respectively.

---

## 3. Class-Level Performance

The final model's performance differs considerably between the three target classes.

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------: | -----: | -------: | ------: |
| Dropout  |    0.8226 | 0.7183 |   0.7669 |     284 |
| Enrolled |    0.4577 | 0.5786 |   0.5111 |     159 |
| Graduate |    0.8509 | 0.8394 |   0.8451 |     442 |

### Dropout

The model achieved:

* Precision: **0.8226**
* Recall: **0.7183**
* F1-score: **0.7669**

The relatively high precision indicates that predictions classified as Dropout were generally reliable.

The model correctly identified approximately 71.83% of the actual Dropout instances.

---

### Enrolled

The Enrolled class produced the weakest performance:

* Precision: **0.4577**
* Recall: **0.5786**
* F1-score: **0.5111**

This indicates that distinguishing students in the Enrolled class was more difficult for the model than distinguishing Dropout or Graduate students.

The lower precision also indicates that a considerable number of students predicted as Enrolled actually belonged to another class.

This class therefore represents the main weakness of the final model.

---

### Graduate

The Graduate class achieved the strongest performance:

* Precision: **0.8509**
* Recall: **0.8394**
* F1-score: **0.8451**

The model correctly identified approximately 83.94% of actual Graduate instances.

The high precision and recall indicate strong predictive performance for this class.

---

## 4. Confusion Matrix

The confusion matrix is:

```text
                  Predicted
               Dropout  Enrolled  Graduate
Actual Dropout    204       52        28
Actual Enrolled    30       92        37
Actual Graduate    14       57       371
```

The corresponding matrix is:

```text
[
    [204, 52,  28],
    [30,  92,  37],
    [14,  57, 371]
]
```

The rows represent the actual classes and the columns represent the predicted classes.

---

## 5. Confusion Matrix Analysis

### Dropout

Out of 284 actual Dropout students:

* 204 were correctly classified as Dropout.
* 52 were classified as Enrolled.
* 28 were classified as Graduate.

Therefore, the largest source of error for Dropout was confusion with the Enrolled class.

### Enrolled

Out of 159 actual Enrolled students:

* 92 were correctly classified as Enrolled.
* 30 were classified as Dropout.
* 37 were classified as Graduate.

The Enrolled class has substantial confusion with both other classes.

### Graduate

Out of 442 actual Graduate students:

* 371 were correctly classified as Graduate.
* 14 were classified as Dropout.
* 57 were classified as Enrolled.

The model performs substantially better at identifying Graduate students than identifying Enrolled students.

---

## 6. Benchmark vs Final Evaluation

The selected Random Forest achieved a cross-validation Macro F1 of:

**0.7165**

The final model achieved a test Macro F1 of:

**0.7077**

The difference is approximately:

**0.0088**

Similarly, the cross-validation Balanced Accuracy was **0.7113**, while the final test Balanced Accuracy was **0.7121**.

| Metric            | Cross-Validation |   Test |
| ----------------- | ---------------: | -----: |
| Macro F1          |           0.7165 | 0.7077 |
| Balanced Accuracy |           0.7113 | 0.7121 |

The final test performance is close to the cross-validation benchmark, indicating that the final model maintained broadly similar performance on the unseen test dataset.

---

## 7. Key Findings

The final evaluation produces several important observations.

### Overall performance

The model achieves approximately **75.37% accuracy** on unseen test data.

### Class-balanced performance

The Macro F1 score is **0.7077**, which provides a more representative view of performance across the three classes than accuracy alone.

### Strongest class

The **Graduate** class is the easiest class for the model to identify, with an F1-score of **0.8451**.

### Weakest class

The **Enrolled** class is the most difficult class, with an F1-score of **0.5111**.

### Dropout prediction

The model achieves an F1-score of **0.7669** for Dropout, with a recall of **0.7183**.

This is particularly relevant because identifying students who eventually drop out is one of the primary objectives of the project.

---

## 8. Final Evaluation Artifacts

The evaluation pipeline stores the following results:

* Overall evaluation metrics
* Classification report
* Confusion matrix

These results are saved as an evaluation artifact and provide a reproducible record of the final model's performance on the test dataset.

---

## 9. Conclusion

The final model demonstrates useful predictive performance across the three academic outcome classes, achieving a Macro F1 of **0.7077** and Balanced Accuracy of **0.7121** on the untouched test dataset.

The model performs particularly well for the **Graduate** and **Dropout** classes, while the **Enrolled** class remains considerably more difficult to distinguish.

Therefore, the final evaluation demonstrates that the model has learned meaningful patterns from the student data, while also highlighting the class-level limitations that should be considered when using the model for academic-risk identification.
