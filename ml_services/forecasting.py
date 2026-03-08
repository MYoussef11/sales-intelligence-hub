import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine
import os
import sys
from datetime import timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def get_sales_data(dealer_id=None):
    engine = create_engine(settings.DATABASE_URL)
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
    Create time-series-aware features for XGBoost forecasting.
    Includes lag features, rolling means, and calendar features.
    """
    df = df.copy()
    
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    
    # Lag features (prior sales values)
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df['sale_price'].shift(lag)
    
    # Rolling mean features (smoothed trends)
    df['rolling_mean_7'] = df['sale_price'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_mean_30'] = df['sale_price'].shift(1).rolling(window=30, min_periods=1).mean()
        
    return df

FEATURE_COLS = [
    'day_of_week', 'month', 'year', 'day_of_year', 'quarter',
    'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30'
]

def train_forecast_model(dealer_id=None):
    logger.info(f"Training XGBoost forecast model for dealer_id={dealer_id}...")
    df = get_sales_data(dealer_id)
    
    if df.empty:
        logger.warning(f"No data found for dealer_id={dealer_id}")
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
    df_features = df_features.dropna()
    
    X = df_features[FEATURE_COLS]
    y = df_features['sale_price']
    
    # Train/Test Split (80/20 time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info("=" * 50)
    logger.info(f"FORECASTING — Evaluation (Dealer {dealer_id})")
    logger.info("=" * 50)
    logger.info(f"  RMSE: {rmse:,.2f}")
    logger.info(f"  MAE:  {mae:,.2f}")
    logger.info(f"  Train Size: {len(X_train)} days")
    logger.info(f"  Test Size:  {len(X_test)} days")
    logger.info("-" * 50)
    
    # Retrain on full data for production forecasting
    model.fit(X, y)
    
    # Forecast next 30 days with iterative prediction
    last_known = df.copy()
    future_dates = [df['date'].max() + timedelta(days=x) for x in range(1, 31)]
    
    predictions = []
    for future_date in future_dates:
        # Create a row for the future date
        new_row = pd.DataFrame({'date': [future_date], 'sale_price': [0]})
        last_known = pd.concat([last_known, new_row], ignore_index=True)
        
        # Recompute features on the extended series
        temp = create_features(last_known)
        row_features = temp.iloc[-1:][FEATURE_COLS]
        
        # Predict and update the sale_price for next iteration
        pred = model.predict(row_features)[0]
        pred = max(pred, 0)  # No negative revenue
        last_known.iloc[-1, last_known.columns.get_loc('sale_price')] = pred
        predictions.append(pred)
    
    result = pd.DataFrame({'date': future_dates, 'forecast': predictions})
    return result, "Success"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forecast, status = train_forecast_model(dealer_id=1)
    if forecast is not None:
        print(forecast.head())
    else:
        print(status)
