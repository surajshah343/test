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

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Pro Dashboard v2.0", layout="wide")

t1, t2 = st.columns([0.9, 0.1])
with t1:
    st.title('🧠 Robust Continuous Learning AI Dashboard')
with t2:
    with st.popover("?"):
        st.markdown("**Updates:** Now uses TimeSeriesSplit to prevent leakage, Log-Return math for consistency, and Empirical CIs from 1,000 Monte Carlo paths.")

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500) # Increased for statistical significance 

MODEL_FILE = f"saved_models/{selected_stock}_v2.json"
META_FILE = f"saved_models/{selected_stock}_meta_v2.json"

# -----------------------------------------------------------------------------
# DATA LOADING (Transition to Log-Returns)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    try:
        data = yf.download(ticker, start=start_date, end=TODAY)
        if data is None or data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        # Calculate Log Returns for mathematical consistency [6, 2]
        data = np.log(data['Close'] / data['Close'].shift(1))
        return data].dropna().copy()
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

data = load_data(selected_stock, START)
if data is None or data.empty: st.stop()

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING (Stationary Features)
# -----------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    # Use Log Returns as the base for features to improve stationarity [1, 7]
    df = df['Close'].rolling(10).mean() / df['Close'] - 1
    df = df['Close'].rolling(20).mean() / df['Close'] - 1
    df['Vol_20'] = df.rolling(20).std()
    
    # RSI on Log Returns
    delta = df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df = 100 - (100 / (1 + rs))
    
    # Target: Residual Alpha over 50-day rolling drift
    df = df.rolling(50).mean()
    df = df.shift(-1) - df
    
    df = df.dt.dayofyear / 366.0 # Scale periodic features
    df['Month'] = df.dt.month / 12.0
    return df.dropna()

ml_ready = engineer_features(data)
features =
target = 'Target_Residual'
X, y = ml_ready[features], ml_ready[target]

# -----------------------------------------------------------------------------
# TRAINING (TimeSeriesSplit to Prevent Leakage )
# -----------------------------------------------------------------------------
def train_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    tuner = RandomizedSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions={'max_depth': [8, 9, 10], 'n_estimators': },
        cv=tscv, n_iter=5, n_jobs=-1
    )
    tuner.fit(X, y)
    return tuner.best_estimator_

# Continuous Learning Check
should_train = True
if os.path.exists(META_FILE):
    with open(META_FILE, 'r') as f:
        meta = json.load(f)
        if meta["last_trained_date"] == ml_ready.max().strftime("%Y-%m-%d"):
            should_train = False

if should_train:
    final_model = train_model(X, y)
    final_model.save_model(MODEL_FILE)
    with open(META_FILE, 'w') as f:
        json.dump({"last_trained_date": ml_ready.max().strftime("%Y-%m-%d")}, f)
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

# -----------------------------------------------------------------------------
# FORECAST GENERATION (Empirical Monte Carlo )
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_simulation(_model, _historical_df, _n_days, _n_sims):
    last_row = _historical_df.iloc[-1]
    last_price = last_row['Close']
    
    # Calculate drift and vol from historical log returns [11, 2]
    hist_logs = _historical_df.tail(252)
    mu = hist_logs.mean()
    sigma = hist_logs.std()
    
    # Pre-calculate Drift component (Ito's Lemma adjustment) [6, 12]
    daily_drift_adj = mu - 0.5 * (sigma**2)
    
    all_paths = np.zeros((_n_days, _n_sims))
    current_prices = np.full(_n_sims, last_price)
    
    progress_bar = st.progress(0, text="Generating Stochastic AI Paths...")
    
    for d in range(_n_days):
        # Generate Alpha prediction (using simplified feature updates for speed)
        # In a production environment, re-calculate all technical indicators here
        current_features = pd.DataFrame( + pd.Timedelta(days=d)).dayofyear / 366.0,
            'Month': (last_row + pd.Timedelta(days=d)).month / 12.0
        }])
        
        alpha_pred = _model.predict(current_features)
        
        # Stochastic Update: Log-space addition [2, 13]
        shocks = np.random.normal(0, sigma, _n_sims)
        log_returns = daily_drift_adj + alpha_pred + shocks
        current_prices = current_prices * np.exp(log_returns)
        all_paths[d, :] = current_prices
        
        if d % 50 == 0: progress_bar.progress((d+1)/_n_days)
            
    progress_bar.empty()
    return all_paths

sim_results = run_simulation(final_model, ml_ready, forecast_days, n_simulations)

# Derive Metrics from Paths 
median_forecast = np.median(sim_results, axis=1)
upper_ci = np.percentile(sim_results, 97.5, axis=1)
lower_ci = np.percentile(sim_results, 2.5, axis=1)

# -----------------------------------------------------------------------------
# UI & VISUALIZATION
# -----------------------------------------------------------------------------
st.subheader("📊 Performance & Forecast")
# Discarding MAPE for MAE due to near-zero targets 
test_split = int(len(ml_ready) * 0.8)
test_preds = final_model.predict(ml_ready.iloc[test_split:][features])
mae = mean_absolute_error(ml_ready.iloc[test_split:][target], test_preds)

c1, c2 = st.columns(2)
c1.metric("Daily Alpha MAE", f"{mae:.5f}")
c2.metric("Simulated Paths", n_simulations)

fig = go.Figure()
future_dates = pd.date_range(ml_ready.max(), periods=forecast_days+1, freq='B')[1:]

# Plot Historical
fig.add_trace(go.Scatter(x=ml_ready, y=ml_ready['Close'], name='Historical', line=dict(color='black')))

# Plot Forecast CI and Median
fig.add_trace(go.Scatter(x=future_dates, y=upper_ci, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=lower_ci, line=dict(width=0), fill='tonexty', 
                         fillcolor='rgba(0, 100, 255, 0.2)', name='95% Empirical CI'))
fig.add_trace(go.Scatter(x=future_dates, y=median_forecast, name='AI Median Path', line=dict(color='blue', width=3)))

fig.update_layout(title=f"{selected_stock} AI Stochastic Forecast", template='plotly_white', height=600)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# PRICE TARGET SUMMARY (Milestone Analysis)
# -----------------------------------------------------------------------------
st.markdown("### 🎯 Distribution-Based Price Targets")
current_price = ml_ready['Close'].iloc[-1]
horizons = {"6 Months": 126, "1 Year": 252}

summary_data =
for label, idx in horizons.items():
    if idx < forecast_days:
        p_slice = sim_results[idx-1, :]
        summary_data.append({
            "Horizon": label,
            "Median Target": f"${np.median(p_slice):.2f}",
            "Lower Bound (2.5%)": f"${np.percentile(p_slice, 2.5):.2f}",
            "Upper Bound (97.5%)": f"${np.percentile(p_slice, 97.5):.2f}",
            "Prob. of Gain": f"{(p_slice > current_price).mean()*100:.1f}%"
        })

st.table(pd.DataFrame(summary_data))
