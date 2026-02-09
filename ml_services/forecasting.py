import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sqlalchemy import create_engine
import os
import sys
from datetime import timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.schema import Transaction

DATABASE_URL = "postgresql://admin:admin123@localhost:5432/sales_intelligence"

def get_sales_data(dealer_id=None):
    engine = create_engine(DATABASE_URL)
    query = """
    SELECT date, sale_price 
    FROM transactions 
    """
    if dealer_id:
        query += f" WHERE dealer_id = {dealer_id}"
    
    df = pd.read_sql(query, engine)
    return df

def create_features(df):
    """
    Create time series features for XGBoost
    """
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Lag features
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df['sale_price'].shift(lag)
        
    return df

def train_forecast_model(dealer_id=None):
    print(f"Training XGBoost forecast model for dealer_id={dealer_id}...")
    df = get_sales_data(dealer_id)
    
    if df.empty:
        return None, "No data found"
        
    # Aggregate by day
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.groupby('date')['sale_price'].sum().reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Fill missing dates with 0
    full_idx = pd.date_range(start=df['date'].min(), end=df['date'].max())
    df = df.set_index('date').reindex(full_idx, fill_value=0).rename_axis('date').reset_index()
    
    # Feature Engineering
    df_features = create_features(df)
    df_features = df_features.dropna() # Drop rows with NaNs from lags
    
    X = df_features[['day_of_week', 'month', 'year', 'day_of_year', 'lag_1', 'lag_7', 'lag_30']]
    y = df_features['sale_price']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(X, y)
    
    # Forecast next 30 days
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
    future_df = pd.DataFrame({'date': future_dates})
    
    # Recursive forecasting (simplified: using last known values for lags)
    # In production, we'd update lags iteratively. For POC, we use static recent history.
    future_features = create_features(pd.concat([df.tail(30), future_df])).tail(30)
    # Fill lags with mean/recent values for simplicity in POC
    future_features = future_features.fillna(method='ffill').fillna(0) 
    
    X_future = future_features[['day_of_week', 'month', 'year', 'day_of_year', 'lag_1', 'lag_7', 'lag_30']]
    predictions = model.predict(X_future)
    
    result = pd.DataFrame({'date': future_dates, 'forecast': predictions})
    return result, "Success"

if __name__ == "__main__":
    forecast, status = train_forecast_model(dealer_id=1)
    if forecast is not None:
        print(forecast.head())
    else:
        print(status)
