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

![L2 Logisitc Regression ROC Curve](Customer Churn Analysis/figures/figure1.png)
> The first plot shows the Receiver Operating Characteristic (ROC) curve for the L2 Logistic Model's predictions. The blue curve lies well above the diagonal dashed line, which represents random guessing. This indicates that the model is able to distinguish between the two classes much better than random chance. The steep initial rise of the curve shows that the model can capture a large portion of true positives while keeping false positives low in the early stages. The curve gradually flattens toward the top-right, suggesting that as the model captures nearly all positives, it also starts including more false positives. Overall, the high placement of the curve suggests good predictive performance, with an Area Under the Curve (AUC) of 0.7400, meaning the model is moderately effective at classification.

![L2 Logisitc Regression Calibration Curve](Customer Churn Analysis/figures/figure2.png)
> The second plot shows the calibration curve for the L2 Logistic Model's predicted probabilities. The red points lie close to the diagonal dashed line, indicating that the model’s predicted probabilities generally match the observed frequencies. There is a slight upward deviation at higher probabilities, suggesting the model somewhat overestimates positive outcomes in this range. Overall, the curve shows decent calibration, meaning the model’s probability estimates are reasonably reliable.

### Logistic Regression (L1)

| Metric    | Train   | Test   | Interpretation                       |
|-----------|---------|--------|-------------------------------------|
| Accuracy  | 0.7415  | 0.7368 | Very similar to L2 performance      |
| Recall    | 0.2744  | 0.2821 | Low, but comparable to L2           |
| Precision | 0.6050  | 0.6199 | Moderate precision                  |
| ROC AUC   | 0.7354  | 0.7400 | Similar discrimination as L2        |

![L1 Logisitc Regression ROC Curve](Customer Churn Analysis/figures/figure3.png)
> The third plot displays the ROC curve for the L1 Logistic Model. The blue curve rises above the diagonal random line, demonstrating that the model can distinguish between positive and negative cases better than chance. The initial slope is less steep than the L2 model, indicating a slightly lower ability to capture true positives early with few false positives. Overall, the curve suggests moderate predictive performance, with an AUC reflecting reasonable classification accuracy.

![L1 Logisitc Regression Calibration Curve](Customer Churn Analysis/figures/figure4.png)
> The fourth plot presents the calibration curve for the L1 Logistic Model. The red points follow the diagonal line closely, showing that predicted probabilities are largely in line with actual outcomes. Minor overestimation occurs at very high probabilities, but overall the model provides fairly accurate probability estimates across most ranges.

**Observation:** Both L1 and L2 logistic regression models perform similarly. Train and test scores are close, indicating no strong overfitting.

### Gradient Boosting Classifier

| Metric    | Train   | Test   | Interpretation                                    |
|-----------|---------|--------|--------------------------------------------------|
| Accuracy  | 0.7440  | 0.7360 | Slight improvement in training accuracy         |
| Recall    | 0.2655  | 0.2687 | Low, similar to logistic regression             |
| Precision | 0.6217  | 0.6238 | Moderate precision, slightly higher than logistic regression |
| ROC AUC   | 0.7400  | 0.7389 | Comparable discrimination ability               |

![Gradient Boosting ROC Curve](Customer Churn Analysis/figures/figure5.png)
> The fifth plot shows the ROC curve for the Gradient Boosting model. The blue curve rises sharply above the diagonal, indicating strong discriminative ability. True positives are captured quickly with few false positives at the beginning, and the curve gradually flattens toward the top-right as additional positives are identified. The high AUC suggests strong overall predictive performance.

![Gradient Boosting Calibration Curve](Customer Churn Analysis/figures/figure6.png)
> The sixth plot shows the calibration curve for the Gradient Boosting model. The red points roughly follow the diagonal line, though there is some overestimation at higher predicted probabilities. This suggests the model is fairly well-calibrated overall, with a slight tendency to be overconfident when predicting near-certain positive outcomes.

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
