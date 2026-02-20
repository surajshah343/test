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

st.set_page_config(page_title="AI Quant Pro v4.0", layout="wide")
st.title('🧠 Financial AI: Quantitative Risk & Regime Intelligence')

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="NVDA").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = int(n_years * 252) 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

# BUMPED VERSION TO v4 TO CLEAR STALE FEATURE CACHE
MODEL_FILE = f"saved_models/{ticker_input}_v4.json"

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
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['MA20'] / df['Close']) - 1
    df['Target_Residual'] = df['Log_Ret'].shift(-1) - df['Log_Ret'].rolling(50).mean()
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    return df.dropna().copy()

ml_data = engineer_features(data)
# Locked in feature set for v4
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

# --- 4. ACCURACY & IMPORTANCE ---
test_preds = final_model.predict(test_set[features])
hit_ratio = np.mean(np.sign(test_preds) == np.sign(test_set[target].values)) * 100
importance = dict(zip(features, final_model.feature_importances_))

# --- 5. REGIME SUMMARY & VaR LOGIC ---
st.subheader("📝 AI Regime & Risk Summary")
top_feature = max(importance, key=importance.get)
current_sentiment = "Bullish" if np.mean(test_preds[-10:]) > 0 else "Bearish"

with st.container(border=True):
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.write(f"**Primary Driver:** {top_feature}")
        st.write(f"**Short-term Bias:** {current_sentiment}")
    
    with col_b:
        analysis = f"The model for **{ticker_input}** is currently heavily influenced by **{top_feature}**. "
        if top_feature == 'Vol_20':
            analysis += "This suggests a 'Volatility Regime' where price swings are the main predictor of future drift. "
        elif top_feature == 'SMA_20_Pct':
            analysis += "The AI is focused on 'Mean Reversion', watching how far the price deviates from its 20-day average. "
        elif top_feature == 'DayOfYear':
            analysis += "Historical seasonality is dominating, implying the stock is following its typical annual cycle. "
        
        analysis += f"With a directional hit ratio of **{hit_ratio:.1f}%**, the model shows "
        analysis += "moderate" if hit_ratio < 53 else "high"
        analysis += " confidence in this regime."
        st.write(analysis)

# --- 6. SIMULATION (Moved up to calculate VaR for metrics) ---
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

# Calculate 95% Value at Risk (VaR) from simulation
final_prices = sim_results[-1, :]
initial_price = data['Close'].iloc[-1]
var_95_price = np.percentile(final_prices, 5)
var_95_pct = ((var_95_price - initial_price) / initial_price) * 100

# --- 7. UI METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${initial_price:.2f}")
m2.metric("Backtest Hit Ratio", f"{hit_ratio:.1f}%", help="Accuracy of predicted direction on unseen data.")
m3.metric("95% Horizon VaR", f"{var_95_pct:.1f}%", help=f"There is a 5% historical probability the asset drops below ${var_95_price:.2f} by the end of the forecast period.")
m4.metric("Primary Driver", top_feature, help="The feature that provided the most 'Gain' in reducing prediction error.")

# --- 8. FEATURE IMPORTANCE CHART ---
st.subheader("📊 Feature Intelligence")
importance_df = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())}).sort_values(by='Importance')
fig_imp = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h', marker_color='#2ecc71'))
fig_imp.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=10, b=10))
st.plotly_chart(fig_imp, use_container_width=True)

# --- 9. STOCHASTIC PROJECTION ---
st.subheader(f"🔮 Stochastic Projection ({n_years}Y)")
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=ml_data['Date'].tail(500), y=ml_data['Close'].tail(500), name='Recent History', line=dict(color='black')))
future_dates = pd.date_range(ml_data['Date'].max(), periods=forecast_days + 1, freq='B')[1:]
fig_future.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 97.5, axis=1), line=dict(width=0), showlegend=False))
fig_future.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 5, axis=1), line=dict(width=0), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)', name='95% VaR Boundary'))
fig_future.add_trace(go.Scatter(x=future_dates, y=np.median(sim_results, axis=1), name='AI Median Forecast', line=dict(color='#2ecc71', width=3)))
st.plotly_chart(fig_future, use_container_width=True)
