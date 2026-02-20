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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# --- CONFIGURATION ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Quant Dashboard v3.5", layout="wide")
st.title('🧠 Financial AI: Interactive Quantitative Framework')

# --- SIDEBAR & TICKER STATE ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="NVDA").upper()

# Reset app state if ticker changes
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ticker_input
if st.session_state.last_ticker != ticker_input:
    st.cache_data.clear()
    st.session_state.last_ticker = ticker_input

n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = int(n_years * 252) 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

MODEL_FILE = f"saved_models/{ticker_input}_v3.json"

# --- 1. DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    df = yf.download(ticker, start=start_date, end=TODAY)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Simple Technicals for Features
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                  -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    return df.dropna().copy()

data = load_data(ticker_input, START)
if data is None: st.stop()

# --- 2. TRAIN/TEST SPLIT VISUALIZATION ---
def get_split_data(df):
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df, split_idx

train_data, test_data, split_idx = get_split_data(data)

with st.expander("🔍 View Train/Test Data Split", expanded=False):
    st.write(f"**Total Samples:** {len(data)} | **Training (80%):** {len(train_data)} | **Testing (20%):** {len(test_data)}")
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Close'], name='Training Data (In-Sample)', line=dict(color='blue')))
    fig_split.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name='Testing Data (Out-of-Sample)', line=dict(color='orange')))
    fig_split.add_vline(x=train_data['Date'].iloc[-1], line_dash="dash", line_color="red", annotation_text="Split Point")
    fig_split.update_layout(title="Data Partitioning for AI Training", template="plotly_white", height=400)
    st.plotly_chart(fig_split, use_container_width=True)

# --- 3. ML PREPARATION ---
def engineer_features(df):
    df = df.copy()
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['MA20'] / df['Close']) - 1
    df['Target_Residual'] = df['Log_Ret'].shift(-1) - df['Log_Ret'].rolling(50).mean()
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    return df.dropna().copy()

ml_data = engineer_features(data)
features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear']
target = 'Target_Residual'

# --- 4. TRAINING ---
if not os.path.exists(MODEL_FILE) or retrain_button:
    # Train ONLY on the first 80% to demonstrate no lookahead
    X_train = ml_data.iloc[:split_idx][features]
    y_train = ml_data.iloc[:split_idx][target]
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
    model.fit(X_train, y_train)
    model.save_model(MODEL_FILE)
    final_model = model
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

# --- 5. REACTIVE SIMULATION ---
# Pass _n_days and _n_sims into arguments to force cache update on slider move
@st.cache_data(show_spinner="Running Monte Carlo...")
def run_simulation(_model, _historical_df, _n_days, _n_sims, _ticker):
    last_price = _historical_df['Close'].iloc[-1]
    mu, sigma = _historical_df['Log_Ret'].tail(252).mean(), _historical_df['Log_Ret'].tail(252).std()
    
    all_paths = np.zeros((_n_days, _n_sims))
    current_prices = np.full(_n_sims, last_price)
    
    for d in range(_n_days):
        # Build features for prediction
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': [np.mean(np.log(current_prices/last_price)) / (d+1) if d > 0 else _historical_df['Log_Ret'].iloc[-1]],
            'SMA_20_Pct': [(_historical_df['MA20'].iloc[-1] / current_prices.mean()) - 1],
            'Vol_20': [sigma],
            'DayOfYear': [(datetime.now().timetuple().tm_yday + d) % 366 / 366.0]
        })
        alpha = _model.predict(pred_feat)[0]
        shocks = np.random.normal(mu - 0.5*sigma**2, sigma, _n_sims)
        current_prices *= np.exp(alpha + shocks)
        all_paths[d, :] = current_prices
        
    return all_paths

# Crucial: These variables now drive the cache key
sim_results = run_simulation(final_model, ml_data, forecast_days, n_simulations, ticker_input)

# --- 6. FORECAST VISUALIZATION ---
st.subheader(f"📊 {ticker_input} Projections ({n_years} Year Horizon)")
fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=ml_data['Date'], y=ml_data['Close'], name='Historical', line=dict(color='black')))

# Future dates correctly match forecast_days
future_dates = pd.date_range(ml_data['Date'].max() + pd.Timedelta(days=1), periods=forecast_days, freq='B')
median_p = np.median(sim_results, axis=1)

fig_main.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 97.5, axis=1), line=dict(width=0), showlegend=False))
fig_main.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 2.5, axis=1), line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 123, 255, 0.2)', name='95% Confidence Interval'))
fig_main.add_trace(go.Scatter(x=future_dates, y=median_p, name='AI Median Forecast', line=dict(color='#007BFF', width=3)))

fig_main.update_layout(template="plotly_white", hovermode="x unified", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_main, use_container_width=True)

# Comparison Table
prices_at_end = sim_results[-1, :]
st.write(f"**Growth Probability:** {(prices_at_end > ml_data['Close'].iloc[-1]).mean()*100:.1f}%")
