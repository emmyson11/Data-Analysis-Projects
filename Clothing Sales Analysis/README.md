Clothing Retail Sales Analysis

Author: Emmy Son

Introduction

This project predicts how much a customer spends annually at a clothing store using demographic, body measurement, and behavioral variables.
The dataset includes:

Self-disclosed gender identity (male, female, nonbinary, other)

Current age

Height (cm), waist size (cm), inseam length (cm)

Self-reported salary (in thousands)

Test group participation (monthly coupons)

Months active in rewards program

Number of purchases

Year of data collection

By building linear and polynomial regression models, the goal is to uncover patterns influencing spending. These insights can help personalize marketing campaigns, optimize promotional strategies, and understand how traits relate to spending habits.

Methods

Data Exploration & Cleaning

Removed incomplete records.

Converted gender into multiple binary variables for model use.

Split dataset into training and testing subsets.

Exploratory Data Analysis (EDA)

Used scatter plots, box plots, and bar charts to identify trends.

Explored relationships between spending and factors like gender, salary, and physical measurements.

Modeling

Linear Regression: Quantifies direct relationships between features and spending.

Polynomial Regression: Captures nonlinear relationships and variable interactions.

Evaluation Metrics

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Mean Absolute Percentage Error (MAPE)

R-squared (R²)

Results
Model Performance
Model	Dataset	MSE	MAE	MAPE	R²
Linear Regression	Training	0.4765	0.5453	327.65%	0.5207
	Testing	0.4818	0.5457	643.97%	0.5292
Polynomial Regression	Training	0.1153	0.2711	239.40%	0.8840
	Testing	0.1118	0.2661	202.92%	0.8907

The polynomial regression model explained ~89% of variation in annual spending (vs. ~52% for linear regression) and had lower prediction errors.

Key EDA Insights
1. Test Group Impact on Spending

Customers receiving monthly coupons (test group) spent more annually than control group customers.

2. Salary vs Purchases and Spending

No clear relationship between salary and number of purchases.

Positive relationship between salary and annual spending.

3. Sales Trends by Year

Sales peaked in 2022, making it the most profitable year in the dataset.

4. Physical Attributes and Spending

Customers with height and inseam close to the average spent more annually.

Spending decreased for customers far from average measurements, suggesting sizing challenges.

5. Height vs Salary by Gender

No consistent linear relationship between height and salary across gender groups.

6. Body Measurements by Year
Year	Height Min (cm)	Height Max (cm)	Height Mean (cm)	Waist Min (cm)	Waist Max (cm)	Waist Mean (cm)	Inseam Min (cm)	Inseam Max (cm)	Inseam Mean (cm)
2019	144.0	191.0	167.96	63.0	121.0	80.50	67.0	81.0	73.99
2020	144.0	189.0	168.02	63.0	121.0	80.51	67.0	81.0	74.00
2021	145.0	189.0	168.03	64.0	123.0	80.52	66.0	83.0	73.99
2022	144.0	194.0	168.10	65.0	125.0	80.57	66.0	81.0	74.04
Discussion / Reflection

Salary was the most informative predictor of annual spending.

Physical attributes like height, waist size, and inseam had weaker predictive power than expected.

No consistent height-salary relationship across gender groups.

EDA proved essential for identifying weak and strong predictors before modeling.

Future improvements could include:

Derived features (e.g., "fit ratio" from height, waist, inseam).

Customer segmentation using clustering.

Incorporating external socioeconomic data (occupation, zip code).
