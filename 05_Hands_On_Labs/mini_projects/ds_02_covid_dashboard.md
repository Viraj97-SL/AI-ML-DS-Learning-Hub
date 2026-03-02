# COVID-19 Interactive Dashboard

> **Difficulty:** Beginner | **Time:** 1-2 days | **Track:** Data Science

## What You'll Build
An interactive COVID-19 dashboard using real Our World in Data (OWID) data. Users can select countries, date ranges, and metrics to compare pandemic trajectories, visualize vaccination progress, and explore correlations.

## Learning Objectives
- Fetch and clean real-world public health data
- Build interactive visualizations with Plotly Express
- Create a multi-page Streamlit dashboard
- Compute rolling averages and per-capita metrics
- Add map visualizations

## Prerequisites
- Basic pandas and Python
- Some familiarity with data visualization

## Tech Stack
- `pandas`: data loading and cleaning
- `plotly.express`: interactive charts and maps
- `streamlit`: dashboard framework
- `requests`: fetching live data

## Step-by-Step Guide

### Step 1: Fetch and Load OWID Data
```python
import pandas as pd
import requests

# OWID provides a constantly updated CSV
URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
df = pd.read_csv(URL, parse_dates=['date'])

# Key columns
COLS = ['iso_code', 'continent', 'location', 'date', 'new_cases', 'new_deaths',
        'total_cases', 'total_deaths', 'people_vaccinated_per_hundred',
        'population', 'total_cases_per_million']
df = df[COLS].dropna(subset=['location', 'date'])

# Remove aggregates (OWID includes world, continents, etc.)
df = df[~df['iso_code'].str.startswith('OWID')]
print(f"Loaded: {df.shape[0]:,} rows | {df['location'].nunique()} countries")
```

### Step 2: Compute Metrics
```python
def compute_rolling_average(df: pd.DataFrame, col: str, window: int = 7) -> pd.DataFrame:
    df = df.sort_values(['location', 'date'])
    df[f'{col}_7day_avg'] = df.groupby('location')[col].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return df

df = compute_rolling_average(df, 'new_cases')
df = compute_rolling_average(df, 'new_deaths')

# Per-million for fair comparison
df['new_cases_per_million'] = df['new_cases'] / df['population'] * 1_000_000
```

### Step 3: Visualize Trends
```python
import plotly.express as px

COUNTRIES = ['United States', 'United Kingdom', 'Germany', 'India', 'Brazil']
subset = df[df['location'].isin(COUNTRIES)]

# Line chart: 7-day average new cases
fig = px.line(subset, x='date', y='new_cases_7day_avg', color='location',
              title='7-Day Average New COVID-19 Cases',
              labels={'new_cases_7day_avg': 'Cases (7-day avg)', 'location': 'Country'})
fig.show()

# Vaccination progress
latest = df.sort_values('date').groupby('location').tail(1)
fig2 = px.bar(latest.sort_values('people_vaccinated_per_hundred', ascending=False).head(20),
              x='location', y='people_vaccinated_per_hundred',
              title='Vaccination Rate (% population)',
              color='continent')
fig2.show()
```

### Step 4: World Map
```python
fig_map = px.choropleth(
    latest,
    locations='iso_code',
    color='total_cases_per_million',
    hover_name='location',
    color_continuous_scale='Reds',
    title='Total COVID Cases per Million People'
)
fig_map.show()
```

### Step 5: Streamlit Dashboard
```python
# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='COVID-19 Dashboard', layout='wide')
st.title('COVID-19 Global Dashboard')

# Sidebar filters
df = pd.read_csv('covid_data.csv', parse_dates=['date'])
countries = st.multiselect('Select Countries', df['location'].unique(), default=['United States', 'Germany'])
date_range = st.date_input('Date Range', [df['date'].min(), df['date'].max()])

filtered = df[(df['location'].isin(countries)) & (df['date'].between(*[pd.Timestamp(d) for d in date_range]))]

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.line(filtered, x='date', y='new_cases_7day_avg', color='location'))
with col2:
    st.plotly_chart(px.line(filtered, x='date', y='new_deaths_7day_avg', color='location'))

# Run: streamlit run dashboard.py
```

## Expected Output
- Multi-panel interactive dashboard with time-series charts
- World choropleth map
- Country comparison charts (normalized per million)
- Date range and country selector controls

## Stretch Goals
- [ ] Add a "Wave Detection" algorithm that automatically identifies COVID waves for each country
- [ ] Build a simple ARIMA or Prophet forecast for the next 30 days of cases
- [ ] Add a correlation scatter plot: cases vs. vaccination rate, GDP, population density

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
