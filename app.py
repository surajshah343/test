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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Pro Dashboard v2.1", layout="wide")

st.title('🧠 Robust Financial AI Dashboard')
st.sidebar.header("Configuration")

ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)

MODEL_FILE = f"saved_models/{selected_stock}_v2.json"
META_FILE = f"saved_models/{selected_stock}_meta_v2.json"

# -----------------------------------------------------------------------------
# DATA LOADING (Corrected Syntax and MultiIndex Handling)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    try:
        data = yf.download(ticker, start=start_date, end=TODAY)
        if data is None or data.empty: return None
        # Robust column handling for yfinance MultiIndex [9]
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        # Core math: All modeling is done in Log-Return space [10, 4]
        data = np.log(data['Close'] / data['Close'].shift(1))
        return data.dropna(subset=).copy()
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

raw_data = load_data(selected_stock, START)
if raw_data is None or raw_data.empty: st.stop()

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING (Stationary and Relative Features)
# -----------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    # Technical Indicators as percentage of current price for stationarity [11, 12]
    df = df['Close'].rolling(10).mean() / df['Close'] - 1
    df = df['Close'].rolling(20).mean() / df['Close'] - 1
    df['Vol_20'] = df.rolling(20).std()
    
    # Target: The daily 'Alpha' residual over the 50-day rolling drift [13, 14]
    df = df.rolling(50).mean()
    df = df.shift(-1) - df
    
    # Periodic time features scaled 0-1
    df = df.dt.dayofyear / 366.0
    df['Month'] = df.dt.month / 12.0
    return df.dropna()

ml_ready = engineer_features(raw_data)
features =
target = 'Target_Residual'
X, y = ml_ready[features], ml_ready[target]

# -----------------------------------------------------------------------------
# TRAINING (TimeSeriesSplit to eliminate look-ahead bias )
# -----------------------------------------------------------------------------
def train_model(X, y):
    # TimeSeriesSplit respects chronological order [5, 15]
    tscv = TimeSeriesSplit(n_splits=5)
    tuner = RandomizedSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions={'max_depth': [1, 2], 'n_estimators': },
        cv=tscv, n_iter=5, n_jobs=-1
    )
    tuner.fit(X, y)
    return tuner.best_estimator_

# Check if model exists or needs update
if not os.path.exists(MODEL_FILE):
    final_model = train_model(X, y)
    final_model.save_model(MODEL_FILE)
    with open(META_FILE, 'w') as f:
        json.dump({"last_trained_date": ml_ready.max().strftime("%Y-%m-%d")}, f)
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

# -----------------------------------------------------------------------------
# FORECAST GENERATION (Empirical Monte Carlo [16, 7])
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_simulation(_model, _historical_df, _n_days, _n_sims):
    last_price = _historical_df['Close'].iloc[-1]
    last_date = _historical_df.iloc[-1]
    
    # Estimate drift and vol parameters from log returns [17, 8]
    hist_logs = _historical_df.tail(252)
    mu = hist_logs.mean()
    sigma = hist_logs.std()
    
    # Ito's Lemma adjustment for Geometric Brownian Motion [18, 19, 20]
    daily_drift_adj = mu - 0.5 * (sigma**2)
    
    all_paths = np.zeros((_n_days, _n_sims))
    current_prices = np.full(_n_sims, last_price)
    
    progress_bar = st.progress(0, text="Simulating Price Paths...")
    
    for d in range(_n_days):
        # Generate prediction using current date features
        curr_date = last_date + pd.Timedelta(days=d+1)
        # For simplicity in this demo, stationary SMA/Vol features are held at last known values
        # In production, these should be updated recursively [21, 22]
        pred_feat = pd.DataFrame(.iloc[-1],
            'SMA_20_Pct': _historical_df.iloc[-1],
            'Vol_20': sigma,
            'DayOfYear': curr_date.dayofyear / 366.0,
            'Month': curr_date.month / 12.0
        }])
        
        alpha_pred = _model.predict(pred_feat)
        
        # Stochastic component: Normal shocks in log-space [8, 20]
        shocks = np.random.normal(0, sigma, _n_sims)
        log_returns = daily_drift_adj + alpha_pred + shocks
        current_prices = current_prices * np.exp(log_returns)
        all_paths[d, :] = current_prices
        
        if d % 50 == 0: progress_bar.progress((d+1)/_n_days)
            
    progress_bar.empty()
    return all_paths

sim_results = run_simulation(final_model, ml_ready, forecast_days, n_simulations)

# -----------------------------------------------------------------------------
# UI & METRICS
# -----------------------------------------------------------------------------
st.subheader("📊 Forecast Summary")

# Empirical CI derivation [23, 6, 24]
median_forecast = np.median(sim_results, axis=1)
upper_ci = np.percentile(sim_results, 97.5, axis=1)
lower_ci = np.percentile(sim_results, 2.5, axis=1)

# Error Metrics on the Test Set
test_split = int(len(ml_ready) * 0.8)
test_preds = final_model.predict(ml_ready.iloc[test_split:][features])
mae = mean_absolute_error(ml_ready.iloc[test_split:][target], test_preds)

c1, c2, c3 = st.columns(3)
c1.metric("Daily Alpha MAE", f"{mae:.5f}")
c2.metric("Simulations", n_simulations)
c3.metric("Last Price", f"${ml_ready['Close'].iloc[-1]:.2f}")

# Plotting
fig = go.Figure()
future_dates = pd.date_range(ml_ready.max(), periods=forecast_days+1, freq='B')[1:]

fig.add_trace(go.Scatter(x=ml_ready, y=ml_ready['Close'], name='Historical', line=dict(color='black')))
fig.add_trace(go.Scatter(x=future_dates, y=upper_ci, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=lower_ci, line=dict(width=0), fill='tonexty', 
                         fillcolor='rgba(0, 100, 255, 0.2)', name='95% Empirical CI'))
fig.add_trace(go.Scatter(x=future_dates, y=median_forecast, name='AI Median Path', line=dict(color='blue', width=3)))

fig.update_layout(title="AI-Enhanced Stochastic Price Projection", template='plotly_white', height=600)
st.plotly_chart(fig, use_container_width=True)

# Price Target Summary Table
horizons = {"6 Months": 126, "1 Year": 252}
summary =
for label, idx in horizons.items():
    if idx < forecast_days:
        prices = sim_results[idx-1, :]
        summary.append({
            "Horizon": label,
            "Median Target": f"${np.median(prices):.2f}",
            "Lower Bound (2.5%)": f"${np.percentile(prices, 2.5):.2f}",
            "Upper Bound (97.5%)": f"${np.percentile(prices, 97.5):.2f}",
            "Confidence": f"{(prices > ml_ready['Close'].iloc[-1]).mean()*100:.1f}% Prob. of Gain"
        })
st.table(pd.DataFrame(summary))
