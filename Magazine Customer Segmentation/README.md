# Customer Segmentation Analysis

## Tech Stack

**Language & Environment**

- Python
- Google Colab / Jupyter Notebook

**Libraries Used**

- `pandas`, `numpy` – Data manipulation  
- `plotnine` – Data visualization (ggplot-style)  
- `scikit-learn` – Modeling and evaluation  
  - `AgglomerativeClustering`, `StandardScaler`  
  - Dimensionality reduction: `PCA`  

---

## Introduction

The goal of this analysis is to **segment customers and articles** to better understand patterns in user behavior and content preferences. The dataset contains:

**Behavioral Data (Customer Features)**  

- `age`, `current_income`, `length_of_subscription`, `monthly_visits`  
- `gender` indicators (`woman`, `nonbinary`, `other`)  
- Interaction metrics: `prop_ads_clicked_log`, `time_spent_browsing_log`, `longest_read_time_log`  
- Combined metrics: e.g., `income_times_visits`, `age_times_ads`  

**Article Data (Content Features)**  

- Counts of article topics per user: `Stocks`, `Productivity`, `Fashion`, `Celebrity`, `Cryptocurrency`, `Science`, `Technology`, `SelfHelp`, `Fitness`, `AI`  

The objective is to **identify natural groupings** of customers and articles, which can be used for **targeted marketing, content recommendations, and personalized engagement strategies**.

---

## Methods

### Data Preprocessing

- Removed irrelevant columns (e.g., `id`)  
- Scaled numerical behavioral features using **StandardScaler** and **RobustScaler** to test sensitivity to outliers  
- One-hot encoded categorical variables (gender indicators)  
- For articles, each column represented topic counts  

### Clustering Approach

**Behavioral Clustering (Customer Segmentation)**  

- **Model:** Agglomerative Hierarchical Clustering (HAC)  
- **Linkage:** Average  
- **Distance metric:** Euclidean  
- **Preprocessing:** Features standardized (standard vs. robust scaling tested)  
- **Hyperparameters:** Number of clusters chosen based on dendrogram analysis and silhouette score (2 clusters identified)  

**Article Clustering (Content Segmentation)**  

- **Model:** Agglomerative Hierarchical Clustering  
- **Linkage:** Average  
- **Distance metric:** Cosine similarity  
- **Hyperparameters:** Number of clusters determined from dendrogram (5 clusters)  

---

## Results

### Behavioral Clustering Model

Two clusters emerged:

| Cluster | Profile Summary |
|---------|----------------|
| 0       | Large mainstream segment: Moderate age/income, average subscription length, higher engagement metrics. |
| 1       | Smaller distinct segment: Slightly older, higher income, more interactions with content and ads. |

> PCA and dendrogram analysis show that the two groups are well-separated and internally cohesive. Boxplots of each feature confirmed clear differences in engagement and demographic characteristics.

---

### Article Clustering Model

The articles were segmented into five thematic clusters:

| Cluster | Focus | Key Topics |
|---------|-------|------------|
| 0       | Science & Fitness with Lifestyle | Science, Fitness, Fashion, Celebrity |
| 1       | SelfHelp, Productivity & Finance | SelfHelp, Productivity, Stocks, AI |
| 2       | Technology & AI | Technology, AI, Science |
| 3       | Finance & Pop Culture | Cryptocurrency, Celebrity, Stocks |
| 4       | Celebrity & Fashion | Celebrity, Fashion, SelfHelp |

> Boxplots for each topic showed the distribution of topic counts per cluster. Cluster profiles provide actionable insights about which types of content appeal to different customer segments.

---

## Discussion / Reflection

From performing these analyses:

- **Behavioral clustering** revealed two main customer segments: a large mainstream group and a smaller, high-engagement group.  
- **Article clustering** revealed five distinct content themes, showing which topics naturally group together.  
- Using dendrograms and PCA helped validate cluster separation and interpretability.  

**Future Work / Improvements:**  

- Experiment with additional clustering algorithms (e.g., KMeans, DBSCAN) to test robustness  
- Incorporate temporal behavioral data for dynamic segmentation  
- Explore linking customer clusters with content clusters for **personalized recommendations**  
- Apply feature importance and dimensionality reduction for better interpretability  

Overall, this project provides a foundation for **targeted marketing and content personalization**, allowing businesses to better understand customer behavior and preferences.
