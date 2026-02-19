import streamlit as st
import os
import json
from datetime import date
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Pro Dashboard", layout="wide")
st.title('🧠 Directional AI Dashboard')

ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()
n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 0, 50, value=20)

MODEL_FILE = f"saved_models/{selected_stock}_continuous_model.json"

# -----------------------------------------------------------------------------
# DATA & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    data = yf.download(ticker, start_date, TODAY)
    if data is None or data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns]
    data.reset_index(inplace=True)
    return data[['Date', 'Close']].copy()

def engineer_features(df):
    df = df.copy()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                  df['Close'].diff().where(df['Close'].diff() < 0, 0).abs().rolling(14).mean())))
    df['Lag_1_Ret'] = df['Close'].pct_change()
    df['Vol_20'] = df['Lag_1_Ret'].rolling(window=20).std()
    df['Rolling_Drift'] = df['Lag_1_Ret'].rolling(window=50).mean()
    df['Target_Return'] = df['Lag_1_Ret'].shift(-1)
    df['Target_Residual'] = df['Target_Return'] - df['Rolling_Drift']
    return df

data = load_data(selected_stock, START)
all_data_engineered = engineer_features(data)
features = ['Lag_1_Ret', 'RSI', 'Vol_20', 'DayOfYear', 'Month']
target = 'Target_Residual'
full_ml_data = all_data_engineered.dropna(subset=features + [target]).copy()

# -----------------------------------------------------------------------------
# BACKTESTING & DIRECTIONAL ACCURACY
# -----------------------------------------------------------------------------
# Split 80% Train, 20% Test (Sequential)
split_idx = int(len(full_ml_data) * 0.8)
train_df = full_ml_data.iloc[:split_idx]
test_df = full_ml_data.iloc[split_idx:]

model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
model.fit(train_df[features], train_df[target])

# Evaluate on Test Set
test_preds_residual = model.predict(test_df[features])

# 1. Directional Accuracy (Hit Ratio)
actual_direction = np.sign(test_df[target])
predicted_direction = np.sign(test_preds_residual)
hit_ratio = (actual_direction == predicted_direction).mean() * 100

# 2. Price-Based MAPE (The "Real" Accuracy)
# Reconstruct Price: Prev_Close * exp(Drift + Pred_Residual)
test_actual_prices = data.iloc[test_df.index]['Close']
test_prev_prices = data.iloc[test_df.index - 1]['Close'].values
reconstructed_price_pred = test_prev_prices * np.exp(test_df['Rolling_Drift'] + test_preds_residual)
price_mape = mean_absolute_percentage_error(test_actual_prices, reconstructed_price_pred) * 100

st.subheader("📊 Model Integrity Report")
c1, c2, c3 = st.columns(3)
c1.metric("Directional Accuracy", f"{hit_ratio:.2f}%", help="How often the AI correctly guessed the price direction.")
c2.metric("Price Forecast Error (MAPE)", f"{price_mape:.2f}%", help="Average % difference between AI price and Actual price.")
c3.metric("Test Samples", len(test_df))

if hit_ratio < 51:
    st.warning("⚠️ Low Accuracy: The model is currently no better than a coin flip. Try a different ticker.")

# -----------------------------------------------------------------------------
# FORECAST GENERATION
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_forecast(_model, _hist_df, _dates, _num_sims):
    # Use the logic from your previous snippet to iterate daily...
    # (Abbreviated for space, assume same iterative prediction logic as your original)
    pass 

# ... (Include your Plotting and Target Table logic from previous prompt) ...
