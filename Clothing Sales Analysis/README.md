# Clothing Retail Sales Analysis

**Author:** Emmy Son  

## Introduction
This project aims to predict **annual customer spending** at a clothing store using demographic, body measurement, and behavioral variables.  

The dataset includes:  
- **Demographics:** Self-disclosed gender identity (male, female, nonbinary, or other), current age  
- **Physical measurements:** Height (cm), waist size (cm), inseam length (cm)  
- **Income:** Self-reported salary (in thousands)  
- **Promotional participation:** Whether the customer is part of a test group receiving monthly coupons  
- **Engagement metrics:** Months active in the store’s rewards program, number of purchases  
- **Temporal:** Year of data collection  

We built **linear** and **polynomial regression models** to uncover patterns influencing spending.  
The results can help **personalize marketing campaigns**, **optimize promotions**, and **better understand how traits relate to spending habits**.

---

## Methods
### Data Exploration & Cleaning
- Removed incomplete records
- Converted gender into binary variables for model compatibility
- Created **training** and **testing** subsets
- Conducted **EDA** using scatter plots, box plots, and bar charts

### Modeling
1. **Linear Regression**
   - Quantifies how individual features (salary, rewards program time, etc.) impact spending
   - Highly interpretable but limited to linear relationships

2. **Polynomial Regression**
   - Captures **nonlinear effects** and **feature interactions**
   - Detects subtler spending patterns

**Performance Metrics:**  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- Mean Absolute Percentage Error (MAPE)  
- R² Score  

---

## Final Results
**Polynomial regression** outperformed linear regression in all metrics:

| Model                    | Dataset       | MSE     | MAE     | MAPE    | R²     |
|--------------------------|--------------|---------|---------|---------|--------|
| **Linear Regression**    | Training     | 0.4765  | 0.5453  | 327.65% | 0.5207 |
|                          | Testing      | 0.4818  | 0.5457  | 643.97% | 0.5292 |
| **Polynomial Regression**| Training     | 0.1153  | 0.2711  | 239.40% | 0.8840 |
|                          | Testing      | 0.1118  | 0.2661  | 202.92% | 0.8907 |

**Key Insights:**
- Polynomial model explained **~89%** of spending variation (vs. ~52% for linear)
- More complex relationships exist between salary, age, and measurements
- Predictions for **low spenders** remain less reliable

---

## Exploratory Data Analysis (EDA)

### **Q1:** Does being in the “test group” increase spending?
- **Finding:** Customers receiving monthly coupons spend more annually.
- **Figure 1:** Boxplot shows test group > control group spending.

### **Q2:** Does higher salary increase purchases and spending?
- **Purchases:** No clear link between salary and purchase frequency.
- **Spending:** Positive relationship—higher salary → higher annual spend.

### **Q3:** Which year had the highest sales?
- **Finding:** Sales peaked in **2022**.
- **Figure 3:** Bar chart shows total spend by year.

### **Q4:** Does body size affect spending?
- **Inseam:** Customers near average inseam spend more.
- **Height:** Spending highest for customers near average height.

### **Q5:** Is there a relationship between salary and height by gender?
- **Finding:** No clear correlation across genders.

### **Q6:** Have customer body measurements changed over time?
| Year | Height Min | Height Max | Height Mean | Waist Min | Waist Max | Waist Mean | Inseam Min | Inseam Max | Inseam Mean |
|------|-----------:|-----------:|------------:|----------:|----------:|-----------:|-----------:|-----------:|------------:|
| 2019 | 144.0      | 191.0      | 167.96      | 63.0      | 121.0     | 80.50      | 67.0       | 81.0       | 73.99       |
| 2020 | 144.0      | 189.0      | 168.02      | 63.0      | 121.0     | 80.51      | 67.0       | 81.0       | 74.00       |
| 2021 | 145.0      | 189.0      | 168.03      | 64.0      | 123.0     | 80.52      | 66.0       | 83.0       | 73.99       |
| 2022 | 144.0      | 194.0      | 168.10      | 65.0      | 125.0     | 80.57      | 66.0       | 81.0       | 74.04       |

---

## Discussion & Reflection
- **Salary** was the strongest predictor of spending  
- Physical attributes had **less predictive power** than expected  
- No consistent link between **height** and **salary** across genders  
- EDA confirmed the **importance of pattern clarity** before modeling  

**Future Improvements:**
- Create derived features (e.g., “fit ratio” combining height, waist, inseam)  
- Segment customers by body type or price sensitivity (clustering)  
- Incorporate **external socioeconomic data** for better predictions  

---

## Repository Structure
├── Clothing_Sales_Analaysis.ipynb/ # Jupyter notebook for EDA and modeling
├── Clothing Retail Sales Analysis - Report.docx/ #Full Report
└── README.md # Project overview
