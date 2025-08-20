# WiDS Datathon 2025: ADHD & Sex Prediction — EDA, Modeling, and Results

Welcome! This repository documents all **EDA, preprocessing, modeling, and key results** for the WiDS Datathon 2025 challenge to predict **ADHD diagnosis** and **biological sex** from neuroimaging and psychological data.

***

## Project Summary

This project leverages comprehensive neural, psychological, parenting, and demographic data to predict ADHD and sex in children. We showcase the complete pipeline: data merging, cleaning, imputation, feature engineering, exploratory visualization, multiple machine learning models, and thorough results interpretation.

***

## Data Overview

| Split     | Rows  | Columns | Non-Connectome Features | Connectome Features (fMRI) |
|-----------|-------|---------|------------------------|----------------------------|
| Training  | 1,213 | 19,929  | 17,982                 | 1,945–>594 (filtered)      |
| Test      | 1,213 | 19,927  | 17,982                 | 1,945–>594 (filtered)      |

**Targets:**  
- `ADHD_Outcome` (binary)  
- `Sex_F` (binary; 1=Female, 0=Male)

**Features:**  
- fMRI connectomes  
- Strength & Difficulties Questionnaire (SDQ; 9 measures)  
- Alabama Parenting Questionnaire (APQ; 6 measures)  
- Clinical (Handedness, Color Vision)  
- Demographics (Study site, ethnicity, race, parent’s education)

***

## Preprocessing

- **Missing values** handled by mean (numerical) or mode (categorical) imputation.
- **Categorical encoding:** LabelEncoder for categorical features.
- **All features standardized** using `StandardScaler`.
- **Train/validation split:** Typical 80/20 via `train_test_split`.

***

## Exploratory Data Analysis (Visualizations & Interpretations)

### Strength & Difficulties Questionnaire (SDQ)
- **Highest ADHD correlation**: Hyperactivity and conduct problems.
- **Boxplots** show higher SDQ scores for diagnosed ADHD.
- *Example interpretation:* Children scoring high on Hyperactivity and Conduct Problems are at significantly increased ADHD risk.

### Alabama Parenting Questionnaire (APQ)
- **Distributions:**  
  - Most parents report low physical discipline, high engagement, and strong positive reinforcement.
  - Monitoring practices are bimodal—two groups emerge for supervision.
- **Correlations:**  
  - High inconsistent discipline and poor monitoring are positively associated with ADHD symptoms.
  - High positive parenting relates to better outcomes, even among ADHD cases.

### Clinical Measures
- **Handedness:** Wide range, majority right-handed; **no significant ADHD or sex association** (t-tests not significant).
- **Color Vision:** Most have perfect scores; **no significant link** to ADHD or sex (chi-squared not significant).

### Demographics
- **Sites:** Staten Island (majority), Midtown, Harlem, MRV (tiny subsample).
- **Diversity:** Broad race/ethnicity/parental education distribution; possible generalizability issues due to site/sample imbalance.

### Brain Connectivity
- **Connectome values** center around zero, indicating many weak or neutral correlations.
- **ADHD vs. Non-ADHD:** Some connectivity features show meaningful differences; bars above zero mean connection is stronger in ADHD, below zero stronger in controls.
- *Interpretation:* Even a random sample of 10–20 connections can show subtle brain network differences between ADHD and controls.

***

## Modeling Approaches & Results

| Model                 | ADHD F1 | ADHD Accuracy | Sex F1 | Sex Accuracy |
|-----------------------|---------|--------------|--------|--------------|
| Logistic Regression   | 0.807   | 0.712        | 0.537  | 0.667        |
| Random Forest         | 0.846   | 0.733        | 0.000  | 0.704        |
| XGBoost               | 0.859   | 0.782        | 0.305  | 0.700        |
| Neural Network (ADHD) | 0.777   | 0.663        | —      | —            |
| Neural Network (Sex)  | —       | —            | 0.573  | 0.749        |

- **Best ADHD prediction:** XGBoost (F1=0.86), Random Forest and Logistic Regression also strong.
- **Sex prediction:** All classical models struggled (best F1=0.57 with neural nets), suggesting sex is a more subtle or imbalanced classification problem here.

***

## Visualized Patterns & Key Interpretations

- **Model metrics reinforce**: Clinical, family, and psychological patterns observed in EDA drive true model performance—strongest predictors are psychological/parenting and selected connectome features.
- **Feature importances** (not shown numerically here due to PDF format but referenced in modeling): SDQ hyperactivity/conduct, APQ monitoring/discipline, specific connectome regions.
- **Classical and neural models both tried**; overfitting checked via train/validation split and score stability.
- **No evidence for clinical (handedness, color vision) or demographic features being dominant predictors**.
- **Best practices:** Multiple model types, attention to imbalanced sex classification (scale_pos_weight in XGBoost), and neural network regularization.

***

## Usage

1. **Download data:** [Kaggle WiDS Datathon 2025 page](https://www.kaggle.com/competitions/widsdatathon2025/data).
2. **Run notebooks:** In Google Colab or local Jupyter. Data files must be placed per path conventions in the scripts.
3. **Dependencies:** pandas, numpy, scikit-learn, seaborn, matplotlib, lightgbm, xgboost, catboost, tensorflow/keras.

***

## Key Takeaways

- **ADHD prediction is robust**: Certain psychological and parenting measures, and brain network data, can help classify ADHD with high accuracy in youth—sex prediction is harder and needs further investigation.
- **Site and sample diversity, and missing data practices, are essential for reproducible research.**
- **Combining classical ML (XGBoost, Random Forest) and deep learning provides best results; always validate with proper splits.**
- **Interpretations from visual analytics strongly align with model feature importance—demonstrating the value of EDA in neuroscientific ML projects.**
