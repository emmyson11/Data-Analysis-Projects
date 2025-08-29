# Medical Insurance Cost Prediction

## Project Overview
This project builds a **predictive model for medical insurance costs** using demographic and lifestyle features such as age, BMI, smoking status, region, and number of children. The goal is to explore which factors drive healthcare costs and to develop a model that can **reliably estimate charges** for new individuals.

---

## Dataset
- **Source:** Provided dataset of **2,772 records** with 7 columns  
- **Features:**
  - `age` — Age of the insured individual
  - `sex` — Gender (male/female)
  - `bmi` — Body Mass Index
  - `children` — Number of dependents covered by insurance
  - `smoker` — Smoking status (yes/no)
  - `region` — Residential region in the U.S.
- **Target:**
  - `charges` — Annual medical insurance cost in USD

No missing values were present in the dataset.

---

## Exploratory Data Analysis (EDA)
Key insights from data visualization:
- **Age**: Costs rise with age, particularly after age 45.  
- **BMI**: Higher BMI correlates with higher costs, especially for smokers.  
- **Smoker status**: The single strongest predictor of charges; smokers incur significantly higher costs.  
- **Region**: Regional differences exist but are less influential compared to smoking, BMI, and age.  

Example plots include:
- Distribution of age, BMI, and charges  
- Count plots for categorical features (sex, smoker, children, region)  
- Scatterplots of age & BMI vs charges  

---

## Methods
### Preprocessing
- Encoded categorical features (`sex`, `smoker`, `region`) as numeric.  
- Split data into **training (80%)** and **test (20%)** sets.  

### Models
1. **Linear Regression (baseline)**
   - Captures linear relationships between predictors and charges.
2. **Future work (extension):**
   - Regularized regression (Lasso/Ridge)  
   - Tree-based models (Random Forest, XGBoost)  
   - Calibration and prediction intervals for individual-level reliability  

### Evaluation
- **Metric:** R² score (coefficient of determination)  
- Training R² ≈ **0.76**  
- Test R² ≈ **0.73**  

This indicates the model explains ~73% of the variance in medical costs on unseen data.

---

## Results
- Smoking status, BMI, and age were the **top drivers of medical insurance costs**.  
- The baseline model performed well but could be improved with **nonlinear models** and **regularization** to capture complex relationships.  
- Visualization of feature interactions (e.g., age × smoking, BMI × smoking) highlighted high-risk subgroups.  
