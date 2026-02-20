import streamlit as st
import os
import time
import requests
# Set environment variable to fix potential OMP error on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import date, datetime, timedelta
import yfinance as yf
from yahooquery import Ticker as YQTicker
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
st.set_page_config(page_title="AI Quant Pro v12.0 - Bulletproof DL", layout="wide")
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

# --- 1. MULTI-TIER RESILIENT DATA ACQUISITION ---
@st.cache_data(show_spinner=False)
def load_and_prep_data(ticker):
    df = pd.DataFrame()
    
    # Tier 1: yfinance with thread-locking fix
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        # threads=False is crucial to prevent Streamlit Cloud deadlocks
        df = yf.download(ticker, period="10y", session=session, threads=False)
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise ValueError("yfinance blocked.")
        df.reset_index(inplace=True)
        
    except Exception:
        st.toast("Tier 1 (yfinance) blocked. Rerouting to Tier 2 (Yahoo Internal API)...", icon="⚠️")
        
        # Tier 2: yahooquery (Internal JSON API Bypass)
        try:
            yq = YQTicker(ticker)
            df = yq.history(period="10y")
            if df.empty or 'error' in df:
                raise ValueError("yahooquery blocked.")
            df = df.reset_index()
            # Standardize columns to match yfinance format
            df.rename(columns={'date': 'Date', 'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)
            
        except Exception:
            st.toast("Tier 2 completely blocked. Rerouting to Tier 3 (Stooq Server)...", icon="🚨")
            
            # Tier 3: Stooq Data Backup
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*10)
                df = web.DataReader(ticker, 'stooq', start=start_date, end=end_date)
                df = df.sort_index(ascending=True).reset_index()
            except Exception:
                st.error(f"Critical Failure: All 3 data pipelines blocked for {ticker}. The ticker may be invalid or servers are entirely down.")
                return None

    # Standardize 'Date' column name if APIs return slightly different casing
    if 'Date' not in df.columns and 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
        
    # Ensure timezone awareness doesn't break Plotly
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
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

with st.spinner(f"Establishing multi-tier data link for {ticker_input}..."):
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
    cum_strat_returns = (1 + strategy
