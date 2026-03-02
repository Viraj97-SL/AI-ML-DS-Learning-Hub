# Customer Segmentation with K-Means

> **Difficulty:** Intermediate | **Time:** 1-2 days | **Track:** Data Science

## What You'll Build
A customer segmentation analysis using K-Means clustering and PCA visualization. You'll cluster e-commerce customers by purchase behavior, name each segment, and present business-ready insights.

## Learning Objectives
- Apply K-Means clustering to real data
- Use the elbow method to choose optimal K
- Reduce dimensions with PCA for visualization
- Profile and name each cluster meaningfully
- Present data science findings in business terms

## Tech Stack
- `scikit-learn`: K-Means, PCA, preprocessing
- `pandas` / `numpy`: data manipulation
- `matplotlib` / `seaborn`: visualization

## Step-by-Step Guide

### Step 1: Generate RFM Customer Data
```python
import pandas as pd
import numpy as np
np.random.seed(42)

n_customers = 500
customers = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'recency_days': np.random.exponential(30, n_customers).clip(1, 365).astype(int),
    'frequency': np.random.negative_binomial(3, 0.3, n_customers).clip(1, 50),
    'monetary': np.random.lognormal(4.5, 1.2, n_customers).clip(5, 5000).round(2),
    'avg_order_value': np.random.lognormal(3.8, 0.8, n_customers).round(2),
    'num_categories': np.random.randint(1, 8, n_customers),
})
print(customers.describe().round(2))
```

### Step 2: Preprocessing and Elbow Method
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

features = ['recency_days', 'frequency', 'monetary', 'avg_order_value', 'num_categories']
X = customers[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find optimal K
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.axvline(x=4, color='red', linestyle='--', label='Chosen K=4')
plt.legend()
plt.show()
```

### Step 3: Fit K-Means and Assign Segments
```python
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
customers['segment'] = kmeans.fit_predict(X_scaled)

# Profile each segment
segment_profiles = customers.groupby('segment')[features].mean().round(2)
print("\nCluster Profiles:")
print(segment_profiles)

# Name segments based on characteristics
segment_names = {
    segment_profiles['monetary'].idxmax(): 'High-Value Champions',
    segment_profiles['recency_days'].idxmax(): 'At-Risk Customers',
    segment_profiles['frequency'].idxmax(): 'Loyal Regulars',
}
# Default name for remaining cluster
remaining = set(range(K)) - set(segment_names.keys())
for r in remaining:
    segment_names[r] = 'Occasional Buyers'

customers['segment_name'] = customers['segment'].map(segment_names)
print("\nSegment Distribution:")
print(customers['segment_name'].value_counts())
```

### Step 4: PCA Visualization
```python
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
customers['pc1'] = X_pca[:, 0]
customers['pc2'] = X_pca[:, 1]

print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

plt.figure(figsize=(10, 7))
for name, group in customers.groupby('segment_name'):
    plt.scatter(group['pc1'], group['pc2'], label=name, alpha=0.6, s=50)
plt.legend()
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments — PCA Projection')
plt.show()
```

### Step 5: Business Recommendations
```python
recommendations = {
    'High-Value Champions': 'Offer VIP loyalty rewards, early access to new products, personal account manager.',
    'Loyal Regulars': 'Upsell premium products, offer subscription plans, reward referrals.',
    'At-Risk Customers': 'Send re-engagement emails with discount offers, ask for feedback on past experience.',
    'Occasional Buyers': 'Send targeted promotions based on browsing history, bundle offers.',
}

print("\n=== BUSINESS RECOMMENDATIONS ===")
for segment, count in customers['segment_name'].value_counts().items():
    pct = count / len(customers)
    print(f"\n{segment} ({count} customers, {pct:.0%}):")
    print(f"  → {recommendations.get(segment, 'Needs further analysis')}")
```

## Expected Output
- Elbow curve showing optimal K selection
- 2D PCA scatter plot with color-coded segments
- Segment profiles table (mean RFM metrics per cluster)
- Business recommendation report per segment

## Stretch Goals
- [ ] Compare K-Means to DBSCAN and Hierarchical Clustering — do they produce different, more meaningful segments?
- [ ] Add silhouette score analysis to quantitatively validate the optimal K
- [ ] Build a Streamlit app where marketing can filter by segment and export customer lists

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
