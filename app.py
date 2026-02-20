import streamlit as st
import os
import json
from datetime import date, datetime
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Quant Pro v5.1", layout="wide")
st.title('🧠 Financial AI: Full-History Quantitative Framework')

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=3)
forecast_days = int(n_years * 252) 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=1000)
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

# Version bump to v5.1 to ensure clean model state
MODEL_FILE = f"saved_models/{ticker_input}_v5_1.json"

# --- 1. DATA LOADING & TECHNICALS ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, start=START, end=TODAY)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    
    # Base Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Technical Indicators for UI and Features
    df['MA20'] = df['Close'].rolling(20).mean()
    df['stddev'] = df['Close'].rolling(20).std()
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # Bollinger Bands
    df['Upper'] = df['MA20'] + (df['stddev'] * 2)
    df['Lower'] = df['MA20'] - (df['stddev'] * 2)
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    return df.dropna().copy()

data = load_data(ticker_input)
if data is None: st.stop()

# --- 2. STATISTICALLY RIGOROUS FEATURE ENGINEERING ---
def engineer_features(df):
    df = df.copy()
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['MA20'] / df['Close']) - 1
    
    # Target: Tomorrow's return minus the current 50-day drift (No look-ahead bias)
    df['Target_Residual'] = df['Log_Ret'].shift(-1) - df['Log_Ret'].rolling(50).mean()
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    return df.dropna().copy()

ml_data = engineer_features(data)
features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear']
target = 'Target_Residual'

# Strict Temporal Split to prevent Data Leakage
split_idx = int(len(ml_data) * 0.8)
train_set = ml_data.iloc[:split_idx]
test_set = ml_data.iloc[split_idx:]

# --- 3. TRAINING ---
if not os.path.exists(MODEL_FILE) or retrain_button:
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
    model.fit(train_set[features], train_set[target])
    model.save_model(MODEL_FILE)
    final_model = model
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

# --- 4. ACCURACY & VAR METRICS ---
test_preds = final_model.predict(test_set[features])
hit_ratio = np.mean(np.sign(test_preds) == np.sign(test_set[target].values)) * 100
importance = dict(zip(features, final_model.feature_importances_))

# --- REWRITTEN SIMULATION WITH ROLLING MEMORY ---
@st.cache_data(show_spinner="Simulating Stochastic Paths...")
def run_simulation(_model, _historical_df, n_days, n_sims, ticker):
    last_price = _historical_df['Close'].iloc[-1]
    
    long_term_mu = _historical_df['Log_Ret'].tail(252).mean()
    sigma = _historical_df['Log_Ret'].tail(252).std()
    
    all_paths = np.zeros((n_days, n_sims))
    current_prices = np.full(n_sims, last_price)
    
    # State tracking for dynamic features
    simulated_history = list(_historical_df['Close'].tail(20).values)
    current_log_ret = _historical_df['Log_Ret'].iloc[-1]
    
    for d in range(n_days):
        # Calculate dynamic moving average based on the median simulated path
        current_ma20 = np.mean(simulated_history[-20:])
        current_mean_price = np.mean(current_prices)
        
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': [current_log_ret],
            'SMA_20_Pct': [(current_ma20 / current_mean_price) - 1], 
            'Vol_20': [sigma],
            'DayOfYear': [(datetime.now().timetuple().tm_yday + d) % 366 / 366.0]
        })
        alpha = _model.predict(pred_feat)[0]
        
        shocks = np.random.normal(0, sigma, n_sims)
        log_returns = alpha + long_term_mu + shocks
        current_prices *= np.exp(log_returns)
        all_paths[d, :] = current_prices
        
        # Update rolling state for the next day's prediction
        current_log_ret = np.mean(log_returns)
        simulated_history.append(current_mean_price)
        
    return all_paths

sim_results = run_simulation(final_model, ml_data, forecast_days, n_simulations, ticker_input)

# Calculate 95% Value at Risk (VaR)
final_prices = sim_results[-1, :]
initial_price = data['Close'].iloc[-1]
var_95_price = np.percentile(final_prices, 5)
var_95_pct = ((var_95_price - initial_price) / initial_price) * 100

# --- 5. TOP LEVEL UI ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${initial_price:.2f}")
m2.metric("Backtest Hit Ratio", f"{hit_ratio:.1f}%", help="Directional Accuracy: How often the AI correctly predicted if the stock would outperform or underperform its trend in the Test Set.")
m3.metric("95% Horizon VaR", f"{var_95_pct:.1f}%", help=f"There is a 5% historical probability the asset drops below ${var_95_price:.2f} by the end of the forecast period.")
m4.metric("Test MAE", f"{mean_absolute_error(test_set[target], test_preds):.5f}", help="Mean Absolute Error on the unseen test data.")

# --- 6. MAIN CHART
