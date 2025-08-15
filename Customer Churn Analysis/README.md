# Customer Churn Prediction Model

## Tech Stack

**Language & Environment**

- Python
- Google Colab

**Libraries Used**

- `pandas`, `numpy` – Data manipulation
- `plotnine` – Data visualization (ggplot-style)
- `scikit-learn` – Modeling and evaluation
  - `LogisticRegression`, `GradientBoostingClassifier`
  - Preprocessing: `StandardScaler`
  - Validation: `train_test_split`
  - Metrics: `accuracy_score`, `recall_score`, `precision_score`, `roc_auc_score`, `roc_curve`

---

## Introduction

The goal of this analysis is to **predict customer churn** for a streaming service based on demographic and behavioral factors. The dataset contains 95,844 records and includes the following features:

- **gender**: Customer’s gender (woman, man, nonbinary, other)  
- **age**: Customer’s age  
- **income**: Customer’s income  
- **monthssubbed**: Number of months subscribed  
- **plan**: Subscription plan (A, B, P)  
- **meanhourswatched**: Average hours watched per month  
- **competitorsub**: Indicator of competitor subscriptions  
- **numprofiles**: Number of user profiles in account  
- **cancelled, downgraded, bundle, kids**: Behavioral flags  
- **longestsession**: Duration of longest session  
- **topgenre, secondgenre**: Favorite genres  
- **churn**: Target variable (0 = retained, 1 = churned)  

The primary objective is to **classify customers who are likely to churn** using both **Logistic Regression** (L1 & L2 regularization) and **Gradient Boosting Classifier**, providing insights for targeted retention strategies.

---

## Methods

### Data Cleaning & Preprocessing

- Examined missing values and dropped rows with missing `age`, `income`, or `cancelled` values  
- One-hot encoded categorical variables (`gender`, `plan`, `topgenre`, `secondgenre`)  
- Standardized numerical features using `StandardScaler` for Logistic Regression  

### Modeling Approach

1. **Train-Test Split:**  
   - 90% training, 10% testing  
2. **Logistic Regression:**  
   - L2 penalty (Ridge)  
   - L1 penalty (Lasso, solver='liblinear')  
   - Predictions and probabilities used for ROC and calibration curves  
3. **Gradient Boosting Classifier:**  
   - 100 estimators, learning rate = 0.1, max_depth = 3  

### Evaluation Metrics

- **Accuracy** – Fraction of correctly classified samples  
- **Recall** – Fraction of actual churners correctly identified  
- **Precision** – Fraction of predicted churners that were correct  
- **ROC AUC** – Model’s ability to distinguish between churners and non-churners  
- Calibration curves – Compare predicted probabilities with actual outcomes  

---

## Results

### Logistic Regression (L2)

| Metric    | Train   | Test   | Interpretation                          |
|-----------|---------|--------|----------------------------------------|
| Accuracy  | 0.7416  | 0.7369 | Correctly classifies ~74% of customers |
| Recall    | 0.2744  | 0.2821 | Captures ~28% of actual churners       |
| Precision | 0.6052  | 0.6203 | Predicted churners are correct ~62% of the time |
| ROC AUC   | 0.7354  | 0.7400 | Decent discrimination ability           |

### Logistic Regression (L1)

| Metric    | Train   | Test   | Interpretation                       |
|-----------|---------|--------|-------------------------------------|
| Accuracy  | 0.7415  | 0.7368 | Very similar to L2 performance      |
| Recall    | 0.2744  | 0.2821 | Low, but comparable to L2           |
| Precision | 0.6050  | 0.6199 | Moderate precision                  |
| ROC AUC   | 0.7354  | 0.7400 | Similar discrimination as L2        |

**Observation:** Both L1 and L2 logistic regression models perform similarly. Train and test scores are close, indicating no strong overfitting.

### Gradient Boosting Classifier

| Metric    | Train   | Test   | Interpretation                                    |
|-----------|---------|--------|--------------------------------------------------|
| Accuracy  | 0.7440  | 0.7360 | Slight improvement in training accuracy         |
| Recall    | 0.2655  | 0.2687 | Low, similar to logistic regression             |
| Precision | 0.6217  | 0.6238 | Moderate precision, slightly higher than logistic regression |
| ROC AUC   | 0.7400  | 0.7389 | Comparable discrimination ability               |

**Observation:** Gradient boosting provides similar performance to logistic regression but slightly improves precision on churn prediction.

---

## Discussion / Reflection

- **Logistic regression and gradient boosting** both provide moderate predictive power  
- **Low recall** indicates that many churners are missed; more feature engineering or alternative modeling may improve this  
- **Precision** is moderate, suggesting predicted churners are somewhat reliable  

**Future work:**

- Incorporate behavioral patterns over time (watch history trends)  
- Test additional ensemble models or hyperparameter tuning  
- Explore feature importance to identify key churn drivers  

Overall, this analysis provides a foundation for **targeted retention strategies**, helping the service identify at-risk customers and optimize engagement.
