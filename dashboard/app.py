import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="STOCKSENSE Dashboard", layout="wide", page_icon="📈")

# Theme Colors
COLORS = {
    'Navy': '#0F172A',
    'Slate': '#1E293B',
    'Azure': '#2563EB',
    'Cyan': '#06B6D4',
    'Emerald': '#22C55E',
    'High': '#EF4444', # Red
    'Medium': '#F59E0B', # Orange
    'Low': '#22C55E'  # Green
}

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        df = pd.read_csv(os.path.join(base_dir, 'reports', 'manager_replenishment_output.csv'))
        master_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'master_cleaned.csv'))
        
        if 'category' in master_df.columns:
            cat_df = master_df[['store_id', 'product_id', 'category']].drop_duplicates()
            df = pd.merge(df, cat_df, on=['store_id', 'product_id'], how='left')
        
        df['7_day_forecast'] = df['7_day_forecast'].fillna(0).round().astype(int)
        df['current_stock'] = df['current_stock'].fillna(0).round().astype(int)
        df['recommended_order'] = df['recommended_order'].fillna(0).round().astype(int)
        df['stockout_probability'] = df['stockout_probability'].fillna(0)
        return df, True
    except Exception as e:
        return str(e), False

@st.cache_data
def load_features():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        df = pd.read_csv(os.path.join(base_dir, 'reports', 'model_feature_importance.csv'))
        return df, True
    except Exception as e:
        return str(e), False

@st.cache_data
def load_metrics():
    # Placeholder for loading actual metrics if files exist
    return None, None

df, success = load_data()
if not success:
    st.error(f"Error loading data: {df}. Ensure intelligence pipeline has run.")
    st.stop()

# --- HEADER ---
st.title("STOCKSENSE")
st.subheader("Intelligent Demand Forecasting & Stock-out Risk Management")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

store_filter = st.sidebar.multiselect("Store", options=df['store_id'].unique(), default=[])
product_filter = st.sidebar.multiselect("Product", options=df['product_id'].unique(), default=[])
if 'category' in df.columns:
    cat_filter = st.sidebar.multiselect("Category", options=df['category'].dropna().unique(), default=[])
else:
    cat_filter = []
risk_filter = st.sidebar.multiselect("Risk Level", options=['HIGH', 'MEDIUM', 'LOW'], default=[])

# Apply filters
filtered_df = df.copy()
if store_filter:
    filtered_df = filtered_df[filtered_df['store_id'].isin(store_filter)]
if product_filter:
    filtered_df = filtered_df[filtered_df['product_id'].isin(product_filter)]
if cat_filter:
    filtered_df = filtered_df[filtered_df['category'].isin(cat_filter)]
if risk_filter:
    filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]

if len(filtered_df) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- TOP KPI CARDS ---
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Store × Product Items", len(filtered_df))
with col2:
    st.metric("High Risk Items", len(filtered_df[filtered_df['risk_level'] == 'HIGH']))
with col3:
    st.metric("Medium Risk Items", len(filtered_df[filtered_df['risk_level'] == 'MEDIUM']))
with col4:
    st.metric("Low Risk Items", len(filtered_df[filtered_df['risk_level'] == 'LOW']))
with col5:
    st.metric("Total Rec. Order", int(filtered_df['recommended_order'].sum()))
with col6:
    st.metric("Avg Stock-out Prob", f"{filtered_df['stockout_probability'].mean()*100:.1f}%")

st.markdown("---")

# --- SECTION 1: RISK OVERVIEW ---
st.header("1. Risk Overview")
col_r1, col_r2 = st.columns(2)

with col_r1:
    risk_counts = filtered_df['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    fig1 = px.pie(risk_counts, values='Count', names='Risk Level', 
                  color='Risk Level', 
                  color_discrete_map={'HIGH': COLORS['High'], 'MEDIUM': COLORS['Medium'], 'LOW': COLORS['Low']},
                  title="Risk Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col_r2:
    fig2 = px.histogram(filtered_df, x='stockout_probability', nbins=20,
                        title="Stock-out Probability Distribution",
                        color_discrete_sequence=[COLORS['Azure']])
    st.plotly_chart(fig2, use_container_width=True)

# --- SECTION 2: MANAGER ACTION TABLE ---
st.header("2. Manager Action Table")

def highlight_risk(val):
    if val == 'HIGH':
        color = COLORS['High']
    elif val == 'MEDIUM':
        color = COLORS['Medium']
    elif val == 'LOW':
        color = COLORS['Low']
    else:
        color = ''
    return f'background-color: {color}; color: white' if color else ''

display_cols = ['store_id', 'product_id', 'current_stock', '7_day_forecast', 'stockout_probability', 'risk_level', 'recommended_order', 'manager_action', 'risk_driver']
table_df = filtered_df[display_cols].copy()
table_df.rename(columns={'risk_driver': 'Key Model Drivers'}, inplace=True)
table_df['stockout_probability'] = (table_df['stockout_probability'] * 100).round(1).astype(str) + '%'

st.dataframe(
    table_df.style.map(highlight_risk, subset=['risk_level']),
    use_container_width=True,
    height=400
)

# --- SECTION 3: HIGH-RISK ITEMS ---
st.header("3. High-Risk Items")
st.subheader("Items requiring immediate attention based on stock-out probability.")
high_risk_df = filtered_df[filtered_df['risk_level'] == 'HIGH'].copy()
if len(high_risk_df) > 0:
    high_risk_df = high_risk_df.sort_values(['stockout_probability', 'recommended_order'], ascending=[False, False])
    high_risk_display = high_risk_df[display_cols].copy()
    high_risk_display.rename(columns={'risk_driver': 'Key Model Drivers'}, inplace=True)
    high_risk_display['stockout_probability'] = (high_risk_display['stockout_probability'] * 100).round(1).astype(str) + '%'
    st.dataframe(high_risk_display.style.map(highlight_risk, subset=['risk_level']), use_container_width=True)
else:
    st.success("No HIGH risk items currently.")

# --- SECTION 4: DEMAND FORECAST ---
st.header("4. Demand vs. Stock Analysis")
chart_df = filtered_df.nlargest(10, 'recommended_order').copy()
chart_df['Store_Prod'] = chart_df['store_id'] + " - " + chart_df['product_id']
chart_df = chart_df.sort_values('recommended_order', ascending=True) # for horizontal bar chart

st.markdown("*Showing top 10 items by recommended order.*")
fig4 = go.Figure()
fig4.add_trace(go.Bar(y=chart_df['Store_Prod'], x=chart_df['current_stock'], name='Current Stock', marker_color=COLORS['Cyan'], orientation='h'))
fig4.add_trace(go.Bar(y=chart_df['Store_Prod'], x=chart_df['7_day_forecast'], name='7-Day Forecast', marker_color=COLORS['Azure'], orientation='h'))
fig4.update_layout(barmode='group', title="Current Stock vs 7-Day Forecast (Top 10)")
st.plotly_chart(fig4, use_container_width=True)

# --- SECTION 5: REPLENISHMENT ---
st.header("5. Replenishment Recommendations")
col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    store_orders = filtered_df.groupby('store_id')['recommended_order'].sum().reset_index()
    store_orders = store_orders.sort_values('recommended_order', ascending=True)
    fig5 = px.bar(store_orders, y='store_id', x='recommended_order', orientation='h', title="Total Recommended Order by Store", color_discrete_sequence=[COLORS['Emerald']])
    st.plotly_chart(fig5, use_container_width=True)

with col_rep2:
    pos_orders = filtered_df[filtered_df['recommended_order'] > 0]
    if len(pos_orders) < 10:
        top_prods = filtered_df.groupby('product_id')['recommended_order'].sum().reset_index().nlargest(10, 'recommended_order')
    else:
        top_prods = pos_orders.groupby('product_id')['recommended_order'].sum().reset_index().nlargest(10, 'recommended_order')
    
    top_prods = top_prods.sort_values('recommended_order', ascending=True)
    fig6 = px.bar(top_prods, x='recommended_order', y='product_id', orientation='h', title="Top 10 Products Requiring Replenishment", color_discrete_sequence=[COLORS['Emerald']])
    st.plotly_chart(fig6, use_container_width=True)

# --- SECTION 6: EXPLAINABILITY ---
st.header("6. Explainability")

feat_df, feat_success = load_features()
if feat_success and not feat_df.empty:
    feat_df['Feature'] = feat_df['Feature'].str.replace('_', ' ').str.title()
    fig7 = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Global Model Feature Importance", color_discrete_sequence=[COLORS['Slate']])
    fig7.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig7, use_container_width=True)
else:
    st.info("Feature importance data not available.")

st.markdown("### Why is this item at risk?")
store_sel = st.selectbox("Select Store", options=filtered_df['store_id'].unique())
prod_options = filtered_df[filtered_df['store_id'] == store_sel]['product_id'].unique()
prod_sel = st.selectbox("Select Product", options=prod_options)

item = filtered_df[(filtered_df['store_id'] == store_sel) & (filtered_df['product_id'] == prod_sel)].iloc[0]

st.markdown(f"**Current stock:** {item['current_stock']} units")
st.markdown(f"**7-day forecast:** {item['7_day_forecast']} units")
st.markdown(f"**Stock-out probability:** {item['stockout_probability'] * 100:.1f}%")
st.markdown(f"**Risk threshold crossed:** {item['risk_level']}")

if feat_success and not feat_df.empty:
    top_f = feat_df.nlargest(2, 'Importance')['Feature'].tolist()
    if len(top_f) == 2:
        st.markdown(f"**Key Model Drivers:** {top_f[0]}, {top_f[1]}")
        st.info("The model places high importance on these features. Feature importance indicates variables used by the model; it does not imply causation.")
    else:
        st.markdown(f"**Key Model Drivers:** {item['risk_driver']}")

# --- OPTIONAL WHAT-IF ANALYSIS ---
st.header("7. What-If Scenario Simulation")
st.markdown("*(Planning simulation — not a retrained ML prediction)*")
scenario_col1, scenario_col2 = st.columns(2)
with scenario_col1:
    demand_increase = st.slider("Festival Demand Increase (%)", min_value=0, max_value=100, value=0, step=5)
with scenario_col2:
    supplier_delay = st.slider("Supplier Delay (Days)", min_value=0, max_value=14, value=0, step=1)

if demand_increase > 0 or supplier_delay > 0:
    scenario_forecast = item['7_day_forecast'] * (1 + demand_increase / 100)
    scenario_prob = min(1.0, item['stockout_probability'] + (supplier_delay * 0.05))
    
    st.write(f"**Baseline Forecast:** {item['7_day_forecast']}")
    st.write(f"**Scenario Forecast:** {scenario_forecast:.0f}")
    
    st.write(f"**Baseline Risk:** {item['stockout_probability']*100:.1f}%")
    st.write(f"**Scenario Risk Estimate:** {scenario_prob*100:.1f}%")
    
    scenario_rec = max(0, scenario_forecast + item['safety_stock'] - item['current_stock'] - item['incoming_stock'])
    st.write(f"**Baseline Recommended Order:** {item['recommended_order']}")
    st.write(f"**Scenario Recommended Order:** {scenario_rec:.0f}")

# --- MODEL & DATA NOTES ---
st.markdown("---")
st.header("Model & Data Notes")

st.markdown("### Demand Forecasting")
st.markdown("Model used: *Metrics not available in reports*")

st.markdown("### Stock-out Prediction")
st.markdown("Model used: *Metrics not available in reports*")
st.markdown("*Note: Stock-out labels are highly imbalanced; accuracy should be interpreted together with Precision, Recall, F1 and ROC-AUC.*")

st.header("Data Limitation")
st.markdown("The available dataset contains limited historical observations for many Store × Product combinations. Therefore, demand forecast metrics should be interpreted as a prototype evaluation rather than production-level performance.")
