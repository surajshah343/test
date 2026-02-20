import streamlit as st
import os
import json
from datetime import date, datetime
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Quant Pro v3.8", layout="wide")
st.title('🧠 Financial AI: Predictive Validation & Feature Intelligence')

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="NVDA").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = int(n_years * 252) 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

MODEL_FILE = f"saved_models/{ticker_input}_v3.json"

# --- 1. DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, start=START, end=TODAY)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    return df.dropna().copy()

data = load_data(ticker_input)
if data is None: st.stop()

# --- 2. ML PREPARATION ---
def engineer_features(df):
    df = df.copy()
    # Feature set
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['MA20'] / df['Close']) - 1
    df['Target_Residual'] = df['Log_Ret'].shift(-1) - df['Log_Ret'].rolling(50).mean()
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    return df.dropna().copy()

ml_data = engineer_features(data)
features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear']
target = 'Target_Residual'

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

# --- 4. ACCURACY & HIT RATIO ---
test_preds = final_model.predict(test_set[features])
hits = np.sign(test_preds) == np.sign(test_set[target].values)
hit_ratio = np.mean(hits) * 100

# --- 5. UI: TOP LEVEL METRICS WITH HOVER INFO ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ticker", ticker_input)
m2.metric(
    "Backtest Hit Ratio", 
    f"{hit_ratio:.1f}%", 
    help="Directional Accuracy: How often the AI correctly predicted if the stock would outperform or underperform its trend. Above 53% is considered strong for financial models."
)
m3.metric(
    "Test MAE", 
    f"{mean_absolute_error(test_set[target], test_preds):.5f}",
    help="Mean Absolute Error: The average 'miss' of the AI in log-return terms. Smaller is better. It represents the noise the AI couldn't account for."
)
m4.metric(
    "Current Price", 
    f"${data['Close'].iloc[-1]:.2f}",
    help="The most recent closing price from Yahoo Finance used as the 'starting line' for the simulation."
)

# --- 6. VISUALIZATION: FEATURE IMPORTANCE ---
st.subheader("📊 Feature Intelligence: What's Driving the Model?")
# Get feature importance from XGBoost
importance_scores = final_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_scores}).sort_values(by='Importance', ascending=True)

fig_imp = go.Figure(go.Bar(
    x=importance_df['Importance'],
    y=importance_df['Feature'],
    orientation='h',
    marker_color='#3498db'
))
fig_imp.update_layout(
    title="Relative Contribution to Prediction",
    xaxis_title="Importance Score (Gain)",
    template="plotly_white",
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_imp, use_container_width=True)

with st.expander("📖 How to read Feature Importance"):
    st.write("""
    - **Lag_1_Ret**: Importance of yesterday's price movement. High scores suggest the stock is 'trendy' or 'mean-reverting'.
    - **SMA_20_Pct**: Distance from the 20-day average. High scores mean the model is looking for 'overextended' prices.
    - **Vol_20**: Recent 20-day volatility. High scores mean risk levels are a primary driver of future returns.
    - **DayOfYear**: Seasonal patterns. High scores suggest the stock has strong historical monthly cycles (e.g., 'January Effect').
    """)

# --- 7. FUTURE SIMULATION ---
@st.cache_data(show_spinner="Simulating Future Paths...")
def run_simulation(_model, _historical_df, n_days, n_sims, ticker):
    last_price = _historical_df['Close'].iloc[-1]
    mu, sigma = _historical_df['Log_Ret'].tail(252).mean(), _historical_df['Log_Ret'].tail(252).std()
    all_paths = np.zeros((n_days, n_sims))
    current_prices = np.full(n_sims, last_price)
    
    for d in range(n_days):
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': [_historical_df['Log_Ret'].iloc[-1] if d==0 else np.mean(np.log(current_prices/last_price))/d],
            'SMA_20_Pct': [(_historical_df['MA20'].iloc[-1] / current_prices.mean()) - 1],
            'Vol_20': [sigma],
            'DayOfYear': [(datetime.now().timetuple().tm_yday + d) % 366 / 366.0]
        })
        alpha = _model.predict(pred_feat)[0]
        shocks = np.random.normal(mu - 0.5*sigma**2, sigma, n_sims)
        current_prices *= np.exp(alpha + shocks)
        all_paths[d, :] = current_prices
    return all_paths

sim_results = run_simulation(final_model, ml_data, forecast_days, n_simulations, ticker_input)

st.subheader(f"🔮 AI-Driven Stochastic Projection ({n_years}Y)")
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=ml_data['Date'].tail(500), y=ml_data['Close'].tail(500), name='Recent History', line=dict(color='black')))
future_dates = pd.date_range(ml_data['Date'].max(), periods=forecast_days + 1, freq='B')[1:]
fig_future.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 97.5, axis=1), line=dict(width=0), showlegend=False))
fig_future.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 2.5, axis=1), line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 123, 255, 0.1)', name='95% CI'))
fig_future.add_trace(go.Scatter(x=future_dates, y=np.median(sim_results, axis=1), name='AI Median Forecast', line=dict(color='#3498db', width=3)))
st.plotly_chart(fig_future, use_container_width=True)
