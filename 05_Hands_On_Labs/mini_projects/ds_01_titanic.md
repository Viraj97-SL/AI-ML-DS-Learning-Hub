# Titanic Survival Analysis

> **Difficulty:** Beginner | **Time:** 1-2 days | **Track:** Data Science

## What You'll Build
A complete data science project on the classic Titanic dataset: exploratory data analysis, feature engineering, multiple ML models, and a Streamlit dashboard that lets users predict survival for custom passengers.

## Learning Objectives
- Perform end-to-end EDA on a real dataset
- Handle missing data and engineer features
- Train and compare multiple classifiers
- Visualize model results and feature importance
- Build an interactive Streamlit app

## Prerequisites
- Basic Python and pandas
- Familiarity with scikit-learn

## Tech Stack
- `pandas`: data manipulation and cleaning
- `matplotlib` / `seaborn`: visualization
- `scikit-learn`: ML models
- `streamlit`: interactive dashboard

## Step-by-Step Guide

### Step 1: Load and Explore the Data
```python
import pandas as pd
import seaborn as sns

# Download from Kaggle or use seaborn's built-in
df = sns.load_dataset('titanic')
print(df.shape, df.dtypes)
print(df.isnull().sum())
print(df['survived'].value_counts(normalize=True))
```

### Step 2: Exploratory Data Analysis
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Survival by sex
df.groupby('sex')['survived'].mean().plot(kind='bar', ax=axes[0,0], title='Survival Rate by Sex')

# Survival by class
df.groupby('pclass')['survived'].mean().plot(kind='bar', ax=axes[0,1], title='Survival Rate by Class')

# Age distribution
df.groupby('survived')['age'].plot(kind='kde', ax=axes[0,2], legend=True, title='Age Distribution')

# Fare vs survival
sns.boxplot(x='survived', y='fare', data=df, ax=axes[1,0])
axes[1,0].set_title('Fare by Survival')

# Correlation heatmap
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, ax=axes[1,1])

plt.tight_layout()
plt.show()
```

### Step 3: Feature Engineering and Cleaning
```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fill missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['fare'] = df['fare'].fillna(df['fare'].median())
    # New features
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0,12,18,35,60,100], labels=['child','teen','adult','middle','senior'])
    # Encode categoricals
    df['sex_enc'] = (df['sex'] == 'male').astype(int)
    df = pd.get_dummies(df, columns=['embarked', 'age_group'], drop_first=True)
    return df

df_clean = engineer_features(df)
```

### Step 4: Train and Compare Models
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = ['pclass', 'sex_enc', 'age', 'fare', 'family_size', 'is_alone']
X = df_clean[FEATURES].fillna(0)
y = df_clean['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    results[name] = {'cv_acc': cv_score, 'test_auc': test_auc}
    print(f'{name}: CV Acc={cv_score:.3f}, Test AUC={test_auc:.3f}')
```

### Step 5: Build Streamlit Dashboard
```python
# app.py
import streamlit as st
import pandas as pd
import pickle

st.title('Titanic Survival Predictor')
st.sidebar.header('Passenger Details')

pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 1, 80, 30)
fare = st.sidebar.slider('Fare ($)', 0, 500, 50)
family_size = st.sidebar.slider('Family Size', 1, 10, 1)

features = pd.DataFrame({
    'pclass': [pclass], 'sex_enc': [1 if sex == 'male' else 0],
    'age': [age], 'fare': [fare], 'family_size': [family_size], 'is_alone': [1 if family_size == 1 else 0]
})

# Load best model
model = pickle.load(open('best_model.pkl', 'rb'))
prob = model.predict_proba(features)[0, 1]

st.metric('Survival Probability', f'{prob:.1%}')
st.progress(prob)
if prob > 0.5:
    st.success('Likely to survive!')
else:
    st.error('Unlikely to survive.')

# Run with: streamlit run app.py
```

## Expected Output
- Jupyter notebook with full EDA (10+ visualizations)
- Model comparison table (accuracy, AUC-ROC, precision, recall)
- Feature importance plot for the best model
- Streamlit app running locally

## Stretch Goals
- [ ] Submit predictions to Kaggle and check your public leaderboard score
- [ ] Add SHAP values to the Streamlit app to explain each prediction
- [ ] Try a neural network (PyTorch MLP) and compare it to tree-based models

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
