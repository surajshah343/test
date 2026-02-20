import streamlit as st
import os
import json
from datetime import date, datetime, timedelta
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

st.set_page_config(page_title="AI Quant Dashboard v3.4", layout="wide")
st.title('🧠 Financial AI: Full-History Quantitative Framework')

# --- SIDEBAR & TICKER STATE ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="NVDA").upper()

if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = ticker_input
if st.session_state.current_ticker != ticker_input:
    st.cache_data.clear()
    st.session_state.current_ticker = ticker_input

n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

MODEL_FILE = f"saved_models/{ticker_input}_v3.json"

# --- 1. DATA LOADING & TECHNICALS ---
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, end=TODAY)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        
        # Log Returns
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # RSI (14 Period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['stddev'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['stddev'] * 2)
        df['Lower'] = df['MA20'] - (df['stddev'] * 2)
        
        return df.dropna().copy()
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

data = load_data(ticker_input, START)
if data is None: st.stop()

# --- 2. FEATURE ENGINEERING ---
def engineer_features(df):
    df = df.copy()
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['Close'].rolling(20).mean() / df['Close']) - 1
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['Target_Residual'] = df['Log_Ret'].shift(-1) - df['Log_Ret'].rolling(50).mean()
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    df['Month'] = df['Date'].dt.month / 12.0
    return df.dropna().copy()

ml_data = engineer_features(data)
features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear', 'Month']
target = 'Target_Residual'

# --- 3. TRAINING & SENTIMENT ---
if not os.path.exists(MODEL_FILE) or retrain_button:
    tscv = TimeSeriesSplit(n_splits=5) 
    tuner = RandomizedSearchCV(xgb.XGBRegressor(), {'max_depth': [3, 6], 'n_estimators': [100]}, cv=tscv)
    tuner.fit(ml_data[features], ml_data[target])
    final_model = tuner.best_estimator_
    final_model.save_model(MODEL_FILE)
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

# --- 4. SIMULATION ---
@st.cache_data(show_spinner=False)
def run_simulation(_model, _historical_df, _n_days, _n_sims, _ticker):
    last_price = _historical_df['Close'].iloc[-1]
    hist_tail = _historical_df['Log_Ret'].tail(252)
    mu, sigma = hist_tail.mean(), hist_tail.std()
    
    all_paths = np.zeros((_n_days, _n_sims))
    alphas = []
    current_prices = np.full(_n_sims, last_price)
    current_log_ret = _historical_df['Log_Ret'].iloc[-1]
    
    for d in range(_n_days):
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': [current_log_ret],
            'SMA_20_Pct': [(_historical_df['Close'].tail(20).mean() / current_prices.mean()) - 1],
            'Vol_20': [sigma],
            'DayOfYear': [(datetime.now().timetuple().tm_yday + d) % 366 / 366.0],
            'Month': [(datetime.now().month + (d // 30) - 1) % 12 / 12.0]
        })
        alpha_pred = _model.predict(pred_feat)[0]
        alphas.append(alpha_pred)
        shocks = np.random.normal(mu - 0.5*sigma**2, sigma, _n_sims)
        current_prices *= np.exp(alpha_pred + shocks)
        all_paths[d, :] = current_prices
        current_log_ret = np.mean(alpha_pred + shocks)
    return all_paths, np.mean(alphas)

sim_results, avg_alpha = run_simulation(final_model, ml_data, forecast_days, n_simulations, ticker_input)

# --- 5. UI DISPLAY ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker_input)
c2.metric("Last Price", f"${ml_data['Close'].iloc[-1]:.2f}")
sentiment = "Bullish" if avg_alpha > 0.0001 else "Bearish" if avg_alpha < -0.0001 else "Neutral"
c3.metric("AI Sentiment", sentiment, f"{avg_alpha*100:.4f}% Alpha/Day")
c4.metric("Data Range", "2015 - Present")

# MAIN FORECAST CHART (Full History Enabled)
fig_main = go.Figure()
# Removed .tail() to show full history from 2015
fig_main.add_trace(go.Scatter(x=ml_data['Date'], y=ml_data['Close'], name='Historical Price', line=dict(color='black', width=1.5)))

future_dates = pd.date_range(ml_data['Date'].max(), periods=forecast_days+1, freq='B')[1:]
fig_main.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 97.5, axis=1), line=dict(width=0), showlegend=False))
fig_main.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 2.5, axis=1), line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.15)', name='95% Confidence Interval'))
fig_main.add_trace(go.Scatter(x=future_dates, y=np.median(sim_results, axis=1), name='AI Median Forecast', line=dict(color='#007BFF', width=2.5)))

fig_main.update_layout(title=f"{ticker_input} Long-Term Price Action & AI Forecast", xaxis_rangeslider_visible=True, template="plotly_white")
st.plotly_chart(fig_main, use_container_width=True)

# TECHNICAL INDICATORS (Focus on last 500 days for clarity)
st.subheader("🛠 Technical Regime Analysis (Last 500 Trading Days)")
tech_view = ml_data.tail(500) # Balanced view: enough history to see trends, but not cluttered
fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])

# Bollinger Bands
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Upper'], line=dict(color='rgba(173, 216, 230, 0.5)'), name='Upper Band'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Lower'], line=dict(color='rgba(173, 216, 230, 0.5)'), fill='tonexty', name='Lower Band'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Close'], line=dict(color='black'), name='Close'), row=1, col=1)

# MACD
colors = ['#26a69a' if x > 0 else '#ef5350' for x in tech_view['MACD_Hist']]
fig_tech.add_trace(go.Bar(x=tech_view['Date'], y=tech_view['MACD_Hist'], name='MACD Hist', marker_color=colors), row=2, col=1)
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['MACD'], name='MACD', line=dict(color='#2196f3')), row=2, col=1)

# RSI
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['RSI'], name='RSI', line=dict(color='#9c27b0')), row=3, col=1)
fig_tech.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
fig_tech.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=3, col=1)

fig_tech.update_layout(height=700, showlegend=False, template="plotly_white")
st.plotly_chart(fig_tech, use_container_width=True)

# Horizons Table
horizons = {"6 Months": 126, "1 Year": 252}
summary_list = []
for label, idx in horizons.items():
    if idx <= forecast_days:
        prices = sim_results[idx-1, :]
        summary_list.append({
            "Horizon": label, "Median Target": f"${np.median(prices):.2f}",
            "95% Range": f"${np.percentile(prices, 2.5):.2f} - ${np.percentile(prices, 97.5):.2f}",
            "Growth Prob": f"{(prices > ml_data['Close'].iloc[-1]).mean()*100:.1f}%"
        })
st.table(pd.DataFrame(summary_list))
