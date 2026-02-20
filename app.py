import streamlit as st
import os
import json
from datetime import date
import yfinance as yf
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# --- CONFIGURATION ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Pro Dashboard v3.0", layout="wide")
st.title('🧠 Financial AI: Log-Return & Stochastic Framework')

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="NVDA").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)

MODEL_FILE = f"saved_models/{ticker_input}_v3.json"
META_FILE = f"saved_models/{ticker_input}_meta_v3.json"

# --- 1. DATA LOADING (Log-Return Integrity) ---
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, end=TODAY)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        # Calculate Log Returns for the model target [6, 16]
        df = np.log(df['Close'] / df['Close'].shift(1))
        return df.dropna(subset=).copy()
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

data = load_data(ticker_input, START)
if data is None: st.stop()

# --- 2. FEATURE ENGINEERING (Stationarity) ---
def engineer_features(df):
    df = df.copy()
    # Lagged returns and technical ratios (stationary features) [17, 18]
    df = df.shift(1)
    df = (df['Close'].rolling(20).mean() / df['Close']) - 1
    df['Vol_20'] = df.rolling(20).std()
    
    # Target: The daily Alpha (Residual) over the 50-day drift 
    df = df.rolling(50).mean()
    df = df.shift(-1) - df
    
    # Periodic time features [19]
    df = df.dt.dayofyear / 366.0
    df['Month'] = df.dt.month / 12.0
    return df.dropna().copy()

ml_data = engineer_features(data)
features =
target = 'Target_Residual'
X, y = ml_data[features], ml_data[target]

# --- 3. TRAINING (TimeSeriesSplit to prevent Leakage) ---
def train_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5) # 
    tuner = RandomizedSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions={'max_depth': [2, 20], 'n_estimators': },
        cv=tscv, n_iter=5, n_jobs=-1
    )
    tuner.fit(X, y)
    return tuner.best_estimator_

if not os.path.exists(MODEL_FILE):
    final_model = train_model(X, y)
    final_model.save_model(MODEL_FILE)
    with open(META_FILE, 'w') as f:
        json.dump({"last_trained": ml_data.max().strftime("%Y-%m-%d")}, f)
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

# --- 4. FORECAST GENERATION (Stochastic Paths) ---
@st.cache_data(show_spinner=False)
def run_simulation(_model, _historical_df, _n_days, _n_sims):
    last_price = _historical_df['Close'].iloc[-1]
    last_log_ret = _historical_df.iloc[-1]
    
    # Parameters for Geometric Brownian Motion 
    hist_tail = _historical_df.tail(252)
    mu = hist_tail.mean()
    sigma = hist_tail.std()
    
    # Ito's Lemma adjustment for drift [9, 21]
    daily_drift_adj = mu - 0.5 * (sigma**2)
    
    all_paths = np.zeros((_n_days, _n_sims))
    current_prices = np.full(_n_sims, last_price)
    
    progress_bar = st.progress(0, text="Simulating Price Dynamics...")
    
    for d in range(_n_days):
        # Build feature vector for the prediction [17, 22]
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': [last_log_ret],
            'SMA_20_Pct': [(_historical_df['Close'].rolling(20).mean().iloc[-1]/_historical_df['Close'].iloc[-1])-1],
            'Vol_20': [sigma],
            'DayOfYear': [(date.today().dayofyear + d) % 366 / 366.0],
            'Month': [(date.today().month + (d // 30)) % 12 / 12.0]
        })
        
        alpha_pred = _model.predict(pred_feat)
        
        # Stochastic Update: log_price + drift + alpha + noise 
        shocks = np.random.normal(0, sigma, _n_sims)
        log_returns = daily_drift_adj + alpha_pred + shocks
        current_prices = current_prices * np.exp(log_returns)
        all_paths[d, :] = current_prices
        
        if d % 50 == 0: progress_bar.progress((d+1)/_n_days)
            
    progress_bar.empty()
    return all_paths

sim_results = run_simulation(final_model, ml_data, forecast_days, n_simulations)

# --- 5. VISUALIZATION & ANALYSIS ---
st.subheader("📊 Forecast Insights")
median_p = np.median(sim_results, axis=1)
upper_ci = np.percentile(sim_results, 97.5, axis=1)
lower_ci = np.percentile(sim_results, 2.5, axis=1)

# Backtest Performance
split = int(len(ml_data) * 0.8)
test_preds = final_model.predict(ml_data.iloc[split:][features])
mae = mean_absolute_error(ml_data.iloc[split:][target], test_preds)
rmse = np.sqrt(mean_squared_error(ml_data.iloc[split:][target], test_preds))

c1, c2, c3 = st.columns(3)
c1.metric("Model MAE (Residual)", f"{mae:.5f}")
c2.metric("Model RMSE (Residual)", f"{rmse:.5f}")
c3.metric("Last Price", f"${ml_data['Close'].iloc[-1]:.2f}")

fig = go.Figure()
future_dates = pd.date_range(ml_data.max(), periods=forecast_days+1, freq='B')[1:]
fig.add_trace(go.Scatter(x=ml_data.tail(252), y=ml_data['Close'].tail(252), name='Historical', line=dict(color='black')))
fig.add_trace(go.Scatter(x=future_dates, y=upper_ci, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=lower_ci, line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.2)', name='95% CI'))
fig.add_trace(go.Scatter(x=future_dates, y=median_p, name='AI Median Path', line=dict(color='blue', width=2)))
st.plotly_chart(fig, use_container_width=True)

# Table Output
horizons = {"6 Months": 126, "1 Year": 252}
summary =
for label, idx in horizons.items():
    if idx < forecast_days:
        prices = sim_results[idx-1, :]
        summary.append({
            "Horizon": label,
            "Median Target": f"${np.median(prices):.2f}",
            "95% Confidence Range": f"${np.percentile(prices, 2.5):.2f} - ${np.percentile(prices, 97.5):.2f}",
            "Prob. of Growth": f"{(prices > ml_data['Close'].iloc[-1]).mean()*100:.1f}%"
        })
st.table(pd.DataFrame(summary))
