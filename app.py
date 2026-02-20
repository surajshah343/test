import streamlit as st
import os
import time
import requests
# Set environment variable to fix potential OMP error on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import date, datetime, timedelta
import yfinance as yf
import pandas_datareader.data as web
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
os.makedirs("saved_models", exist_ok=True)
st.set_page_config(page_title="AI Quant Pro v11.0 - Resilient DL", layout="wide")
st.title('🧠 Financial AI: LSTM Deep Learning Framework & Backtester')

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=3)
forecast_days = int(n_years * 252)
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)
seq_length = st.sidebar.slider('LSTM Lookback Window (Days):', 10, 60, value=20)

MODEL_WEIGHTS_PATH = os.path.join("saved_models", f"{ticker_input}_lstm_weights.pth")
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

# --- 1. RESILIENT DATA ACQUISITION & FEATURE ENGINEERING ---
@st.cache_data(show_spinner=False)
def load_and_prep_data(ticker):
    df = pd.DataFrame()
    
    # Try 1: yfinance with Custom User-Agent Bypass
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        df = yf.download(ticker, period="10y", session=session)
        
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            raise ValueError("yfinance returned an empty dataset.")
            
    except Exception as e:
        st.warning(f"Primary API blocked. Rerouting to Stooq backup server...")
        
        # Try 2: pandas-datareader via Stooq API (No Key Required)
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*10)
            df = web.DataReader(ticker, 'stooq', start=start_date, end=end_date)
            df = df.sort_index(ascending=True) # Stooq returns dates in descending order
        except Exception as e2:
            st.error(f"Critical Failure: Both primary and backup APIs rejected the request for {ticker}. Ensure the ticker is valid.")
            return None

    df.reset_index(inplace=True)
    
    # Standardize 'Date' column name if APIs return slightly different casing
    if 'Date' not in df.columns and 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    
    # --- Feature Engineering ---
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    df['Target'] = df['Log_Ret'].shift(-1)
    
    return df.dropna().copy()

with st.spinner(f"Establishing secure data link for {ticker_input}..."):
    df = load_and_prep_data(ticker_input)
if df is None:
    st.stop()

# --- 2. LSTM MODEL DEFINITION ---
class QuantLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(QuantLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# --- 3. DATA PREPARATION FOR PYTORCH ---
features = ['Log_Ret', 'Vol_20', 'RSI', 'MACD_Hist']
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[['Target']])

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

test_days = 252
split_idx = len(X_seq) - test_days

X_train, y_train = torch.FloatTensor(X_seq[:split_idx]), torch.FloatTensor(y_seq[:split_idx])
X_test, y_test = torch.FloatTensor(X_seq[split_idx:]), torch.FloatTensor(y_seq[split_idx:])

backtest_dates = df['Date'].iloc[-test_days:].reset_index(drop=True)
actual_returns = df['Target'].iloc[-test_days:].reset_index(drop=True)

# --- 4. MODEL INSTANTIATION & PERSISTENCE LOGIC ---
model = QuantLSTM(input_size=len(features), hidden_size=32, num_layers=2, output_size=1)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) 

should_train = False
if retrain_button:
    should_train = True
    st.toast("🔄 Force retrain requested. Starting training process...", icon="🤖")
elif not os.path.exists(MODEL_WEIGHTS_PATH):
    should_train = True
    st.toast(f"📁 No saved weights found for {ticker_input}. Training a new model...", icon="🧠")
else:
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
        model.eval()
        st.toast(f"✅ Successfully loaded pre-trained AI matrix for {ticker_input}!", icon="📂")
    except Exception as e:
        st.warning(f"Could not load weights: {e}. Retraining model.")
        should_train = True

if should_train:
    st.markdown("### 🧠 Training Deep Learning Model...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    funny_phrases = [
        "Consulting the Wall Street bets oracles...",
        "Mining pure alpha from market noise...",
        "Bribing the algorithms...",
        "Herding the GPU clusters...",
        "Extracting the tears of short sellers...",
        "Quantifying the unquantifiable...",
        "Almost there, formatting the profit matrix..."
    ]
    
    epochs = 50 
    for epoch in range(epochs):
        model.train() 
        optimizer.zero_grad() 
        outputs = model(X_train) 
        loss = criterion(outputs, y_train) 
        loss.backward() 
        optimizer.step() 
        
        progress = int(((epoch + 1) / epochs) * 100)
        progress_bar.progress(progress)
        phrase_idx = int((epoch / epochs) * len(funny_phrases))
        status_text.text(f"Epoch {epoch+1}/{epochs} | {funny_phrases[min(phrase_idx, len(funny_phrases)-1)]} Loss: {loss.item():.4f}")
        time.sleep(0.02) 
        
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    status_text.success(f"Training Complete! Weights saved to persistent storage.")
    progress_bar.empty()
    model.eval()

# --- 5. THE BACKTESTING ENGINE ---
@st.cache_data(show_spinner=False)
def run_backtest(_model, _X_test, _actual_returns, _scaler_y):
    _model.eval()
    with torch.no_grad():
        preds_scaled = _model(_X_test).numpy()
    
    preds_raw = _scaler_y.inverse_transform(preds_scaled).flatten()
    positions = np.where(preds_raw > 0, 1, -1)
    strategy_returns = positions * _actual_returns.values
    
    cum_bh_returns = (1 + _actual_returns).cumprod() - 1
    cum_strat_returns = (1 + strategy_returns).cumprod() - 1
    
    win_rate = np.mean(strategy_returns > 0) * 100
    strat_total_ret = cum_strat_returns.iloc[-1] * 100
    bh_total_ret = cum_bh_returns.iloc[-1] * 100
    
    return cum_bh_returns, cum_strat_returns, win_rate, strat_total_ret, bh_total_ret, positions

st.markdown("### ⏳ Running Walk-Forward Backtest...")
bt_progress = st.progress(0)
bt_status = st.empty()

bt_phrases = [
    "Simulating past mistakes...",
    "Hindsight is 20/20, computing...",
    "Liquidating imaginary margin calls...",
    "Counting hypothetical lambos..."
]

for i in range(100):
    bt_progress.progress(i + 1)
    phrase_idx = int((i / 100) * len(bt_phrases))
    bt_status.text(f"Backtesting | {bt_phrases[min(phrase_idx, len(bt_phrases)-1)]}")
    time.sleep(0.01)
    
bt_progress.empty()
bt_status.empty()

bh_curve, strat_curve, win_rate, strat_ret, bh_ret, positions = run_backtest(model, X_test, actual_returns, scaler_y)

# --- 6. UI & DASHBOARD ---
st.markdown("---")
st.subheader("🕵️‍♂️ 1-Year AI Backtest Results (Out-of-Sample)")

b1, b2, b3 = st.columns(3)
b1.metric("Buy & Hold Return", f"{bh_ret:.1f}%", help="Total return from simply holding the asset for the last year.")
b2.metric("AI Strategy Return", f"{strat_ret:.1f}%", delta=f"{strat_ret - bh_ret:.1f}% vs B&H", help="Total return from the AI's daily Long/Short strategy.")
b3.metric("AI Win Rate", f"{win_rate:.1f}%", help="The percentage of days the AI's position resulted in a profit.")

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=backtest_dates, y=bh_curve * 100, name='Buy & Hold (Benchmark)', line=dict(color='#95a5a6', width=2, dash='dot')))
fig_bt.add_trace(go.Scatter(x=backtest_dates, y=strat_curve * 100, name='AI Strategy (Long/Short)', line=dict(color='#2ecc71', width=3)))

for i in range(len(positions) - 1):
    color = "rgba(46, 204, 113, 0.1)" if positions[i] == 1 else "rgba(231, 76, 60, 0.1)"
    fig_bt.add_vrect(x0=backtest_dates[i], x1=backtest_dates[i+1], fillcolor=color, layer="below", line_width=0)

fig_bt.update_layout(
    template="plotly_white", 
    hovermode="x unified", 
    height=450, 
    yaxis_title="Cumulative Return (%)", 
    title="Hypothetical 1-Year Performance vs Benchmark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_bt, use_container_width=True)

# --- 7. STABILIZED FUTURE SIMULATION ---
@st.cache_data(show_spinner=False)
def run_lstm_simulation(_model, base_data, n_days, n_sims, _scaler_X, _scaler_y, _seq_length):
    _model.eval()
    last_price = base_data['Close'].iloc[-1]
    
    max_annual_drift = 0.25 
    daily_drift_cap = max_annual_drift / 252

    last_seq = base_data[features].tail(_seq_length).values
    last_seq_scaled = _scaler_X.transform(last_seq)
    
    all_paths = np.zeros((n_days, n_sims))
    current_prices = np.full(n_sims, last_price)
    historical_vol = base_data['Log_Ret'].std()
    
    with torch.no_grad():
        for d in range(n_days):
            seq_tensor = torch.FloatTensor(last_seq_scaled).unsqueeze(0)
            pred_scaled = _model(seq_tensor).item()
            pred_ret = _scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            pred_ret = np.clip(pred_ret, -daily_drift_cap, daily_drift_cap)
            shocks = np.random.normal(0, historical_vol, n_sims)
            daily_returns = pred_ret + shocks
            
            current_prices = current_prices * np.exp(daily_returns)
            current_prices = np.maximum(current_prices, 0.01)
            all_paths[d, :] = current_prices
            
    return all_paths

with st.spinner(f"Simulating {n_simulations} future paths for {n_years} years..."):
    sim_results = run_lstm_simulation(model, df, forecast_days, n_simulations, scaler_X, scaler_y, seq_length)

median_forecast = np.median(sim_results, axis=1)
upper_95_bound = np.percentile(sim_results, 97.5, axis=1)
lower_95_bound = np.percentile(sim_results, 2.5, axis=1)

st.markdown("---")
st.subheader(f"🔮 Forward-Looking Monte Carlo Projection ({n_years}Y)")

fig_main = go.Figure()
hist_plot = df.tail(252)
fig_main.add_trace(go.Scatter(x=hist_plot['Date'], y=hist_plot['Close'], name='Historical Price', line=dict(color='#2c3e50', width=2)))

future_dates = pd.date_range(df['Date'].max(), periods=forecast_days + 1, freq='B')[1:]

fig_main.add_trace(go.Scatter(x=future_dates, y=upper_95_bound, line=dict(width=0), showlegend=False))
fig_main.add_trace(go.Scatter(x=future_dates, y=lower_95_bound, line=dict(width=0), fill='tonexty', fillcolor='rgba(41, 128, 185, 0.2)', name='95% Confidence Interval'))
fig_main.add_trace(go.Scatter(x=future_dates, y=median_forecast, name='AI Median Trajectory', line=dict(color='#e74c3c', width=3)))

fig_main.update_layout(
    template="plotly_white", 
    hovermode="x unified", 
    height=600, 
    yaxis_title="Asset Price ($)",
    title=f"{ticker_input} Price Forecast",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_main, use_container_width=True)
