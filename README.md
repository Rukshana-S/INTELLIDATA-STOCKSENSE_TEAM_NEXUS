# STOCKSENSE

### Predict Demand. Prevent Stock-outs. Power Better Decisions.

STOCKSENSE is a retail decision-support solution developed for the
IntelliData 2026 Data Science Hackathon.

## Problem

NovaMart Retail Pvt. Ltd. wants to:

- Forecast product demand
- Predict stock-out risk
- Explain the reasons behind risk
- Recommend inventory actions

## Solution Pipeline

Raw Data
→ Data Cleaning
→ Data Integration
→ EDA
→ Statistical Analysis
→ Feature Engineering
→ Demand Forecasting
→ Stock-out Risk Prediction
→ Explainability
→ Recommendation
→ Visualization

## Round 1

Round 1 focuses on:

- Data understanding
- Data quality audit
- Data cleaning
- Data integration
- Daily Store × Product aggregation
- Exploratory Data Analysis
- Statistical reasoning

## Datasets

- Transactions
- Products
- Stores
- Inventory
- External Factors

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Statsmodels
- Matplotlib
- Seaborn
- Tableau

## Project Structure

```text
data/
├── raw/
└── processed/

models/
notebooks/
outputs/
reports/
src/


We can improve this later after the entire project is complete.

---

# 6. NOW create the folders/files

Your VS Code should eventually look like:

```text
📁 STOCKSENSE_TEAM_NEXUS

 ├── 📁 data
 │    ├── 📁 raw
 │    └── 📁 processed
 │
 ├── 📁 models
 │
 ├── 📁 notebooks
 │
 ├── 📁 outputs
 │    └── 📁 charts
 │
 ├── 📁 reports
 │
 ├── 📁 src
 │
 ├── 📄 .gitignore
 ├── 📄 README.md
 └── 📄 requirements.txt
# INTELLIDATA-STOCKSENSE_TEAM_NEXUS

## ROUND 3 — INTELLIGENCE LAYER & DASHBOARD

### 1. Risk classification
We use predictive modeling to classify stock-out risk based on probability:
- **HIGH:** probability >= 0.70
- **MEDIUM:** probability >= 0.40 and < 0.70
- **LOW:** probability < 0.40

### 2. Replenishment logic
- **Recommended Stock:** 7-Day Forecast + Safety Stock
- **Recommended Order:** max(0, Recommended Stock - Current Stock - Incoming Stock)

### 3. Manager actions
Based on risk level and recommended order:
- **HIGH:** "URGENT REORDER" or "MONITOR STOCK"
- **MEDIUM:** "PLAN REORDER" or "MONITOR"
- **LOW:** "NORMAL REORDER" or "NO ACTION"

### 4. Explainability
- Evaluates Top features influencing stock-out predictions using `feature_importances_`.
- Dynamic descriptions on the dashboard describing key drivers behind the high-risk scores.

### 5. Dashboard
- Created an interactive UI using Streamlit and Plotly.
- Dynamic filtering by Store, Product, Category, and Risk Level.
- Detailed KPIs, Visualizations (Risk Distribution, Demand vs. Stock Analysis), Manager Action Table, and high-risk prioritizations.

### 6. What-If analysis
- A scenario simulation tab to see how changes in Festival Demand Increase (%) or Supplier Delay affect Recommended Orders and Risk probabilities dynamically.

### 7. How to run the dashboard
First ensure the intelligence data is generated:
```bash
python dashboard/intelligence.py
```
Then start the application:
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```