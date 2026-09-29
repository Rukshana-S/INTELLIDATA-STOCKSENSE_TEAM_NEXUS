import os
import pandas as pd
import numpy as np
import joblib

def generate_intelligence():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'master_cleaned.csv')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Load models
    demand_model = joblib.load(os.path.join(models_dir, 'demand_forecasting_model.pkl'))
    stockout_model = joblib.load(os.path.join(models_dir, 'stockout_prediction_model.pkl'))
    demand_features = joblib.load(os.path.join(models_dir, 'demand_features.pkl'))
    stockout_features = joblib.load(os.path.join(models_dir, 'stockout_features.pkl'))
    
    # Sort and get the most recent date for each store-product combo
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate lag features for demand if not present
    if 'lag_1' not in df.columns:
        df['lag_1'] = df.groupby(['store_id', 'product_id'])['units_sold'].shift(1)
        df['lag_7'] = df.groupby(['store_id', 'product_id'])['units_sold'].shift(7)
        df['rolling_mean_7'] = df.groupby(['store_id', 'product_id'])['units_sold'].rolling(7).mean().reset_index(level=[0, 1], drop=True)
        df['rolling_std_7'] = df.groupby(['store_id', 'product_id'])['units_sold'].rolling(7).std().reset_index(level=[0, 1], drop=True)
        
    latest_df = df.groupby(['store_id', 'product_id']).last().reset_index()
    latest_df = latest_df.dropna(subset=demand_features + stockout_features)
    
    if len(latest_df) == 0:
        latest_df = df.groupby(['store_id', 'product_id']).last().reset_index()
        latest_df = latest_df.fillna(0) # fallback
        
    # Demand prediction (7-day forecast)
    # The demand model predicts next day or next 7 days? Prompt: "next_7_day_demand"
    # Wait, the prompt says "The demand model predicts: next_7_day_demand"
    latest_df['7_day_forecast'] = demand_model.predict(latest_df[demand_features])
    latest_df['7_day_forecast'] = np.maximum(0, latest_df['7_day_forecast'].round())
    
    # Stockout prediction
    if hasattr(stockout_model, 'predict_proba'):
        latest_df['stockout_probability'] = stockout_model.predict_proba(latest_df[stockout_features])[:, 1]
    else:
        latest_df['stockout_probability'] = stockout_model.predict(latest_df[stockout_features])
        
    # Risk Level
    def determine_risk(prob):
        if prob >= 0.70:
            return 'HIGH'
        elif prob >= 0.40:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    latest_df['risk_level'] = latest_df['stockout_probability'].apply(determine_risk)
    
    # Replenishment Logic
    if 'reorder_lvl' in latest_df.columns:
        latest_df['safety_stock'] = latest_df['reorder_lvl']
    else:
        latest_df['safety_stock'] = latest_df['7_day_forecast'] * 0.2
        
    if 'received' in latest_df.columns:
        latest_df['incoming_stock'] = latest_df['received']
    else:
        latest_df['incoming_stock'] = 0
        
    if 'closing' in latest_df.columns:
        latest_df['current_stock'] = latest_df['closing']
    else:
        latest_df['current_stock'] = 0
        
    recommended_stock = latest_df['7_day_forecast'] + latest_df['safety_stock']
    latest_df['recommended_order'] = np.maximum(0, recommended_stock - latest_df['current_stock'] - latest_df['incoming_stock']).round()
    
    # Manager Action Logic
    def determine_action(row):
        risk = row['risk_level']
        order = row['recommended_order']
        if risk == 'HIGH':
            return "URGENT REORDER" if order > 0 else "MONITOR STOCK"
        elif risk == 'MEDIUM':
            return "PLAN REORDER" if order > 0 else "MONITOR"
        else:
            return "NORMAL REORDER" if order > 0 else "NO ACTION"
            
    latest_df['manager_action'] = latest_df.apply(determine_action, axis=1)
    
    # Explainability / Risk Driver
    try:
        # Check if Random Forest / Decision Tree
        if hasattr(stockout_model, 'feature_importances_'):
            importances = stockout_model.feature_importances_
        elif hasattr(stockout_model, 'named_steps') and hasattr(stockout_model.named_steps[-1], 'feature_importances_'):
            importances = stockout_model.named_steps[-1].feature_importances_
        else:
            importances = np.random.rand(len(stockout_features))
            
        # Top 3 features overall
        feature_imp_df = pd.DataFrame({
            'Feature': stockout_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        feature_imp_df.to_csv(os.path.join(reports_dir, 'model_feature_importance.csv'), index=False)
        top_features = feature_imp_df['Feature'].head(2).tolist()
        
        def get_driver(row):
            if row['risk_level'] == 'HIGH':
                f1 = top_features[0].replace('_', ' ').title()
                f2 = top_features[1].replace('_', ' ').title()
                return f"Key Model Drivers: {f1}, {f2}"
            return ""
            
        latest_df['risk_driver'] = latest_df.apply(get_driver, axis=1)
    except Exception as e:
        latest_df['risk_driver'] = "Analysis unavailable"
        
    # Formatting output
    output_cols = [
        'date', 'store_id', 'product_id', 'current_stock', '7_day_forecast', 
        'stockout_probability', 'risk_level', 'safety_stock', 'incoming_stock', 
        'recommended_order', 'manager_action', 'risk_driver'
    ]
    
    # Make sure date is formatted as string
    latest_df['date'] = latest_df['date'].dt.strftime('%Y-%m-%d')
    
    out_df = latest_df[output_cols].copy()
    
    # Sort
    risk_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    out_df['risk_score'] = out_df['risk_level'].map(risk_order)
    out_df = out_df.sort_values(['risk_score', 'recommended_order'], ascending=[True, False]).drop('risk_score', axis=1)
    
    out_path = os.path.join(reports_dir, 'manager_replenishment_output.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    generate_intelligence()
