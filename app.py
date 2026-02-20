import streamlit as st
import os
import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as go
from plotly.subplots import make_subplots

# --- CONFIGURATION & SIDEBAR (Defined first to prevent NameError) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="AI Quant Pro v14", layout="wide")

st.sidebar.header("🕹️ Control Panel")
ticker_input = st.sidebar.text_input("Ticker Symbol:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 3, value=1)
seq_length = st.sidebar.slider('Lookback Window (Days):', 10, 60, value=30)
train_button = st.sidebar.button("🚀 Run Analysis & AI Training")

st.title(f"🧠 AI Financial Framework: {ticker_input}")

# --- 1. MATHEMATICAL LOGIC & TECHNICAL INDICATORS ---
@st.cache_data
def get_advanced_data(ticker):
    # Fetch 5 years of data for stable Fibonacci and MACD calculation
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    # RSI: Standard Wilder's Smoothing logic
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD: 12-26-9 Standard
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['20STD'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['20STD'] * 2)

    # Fibonacci Retracement (Standard 52-week swing)
    recent_yr = df.tail(252)
    high, low = recent_yr['High'].max(), recent_yr['Low'].min()
    diff = high - low
    df['Fib_100'] = high
    df['Fib_618'] = high - 0.382 * diff
    df['Fib_500'] = high - 0.500 * diff
    df['Fib_382'] = high - 0.618 * diff
    df['Fib_0']   = low

    # Target: Next day Log Return (Stationary Target)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['Target'] = df['Log_Ret'].shift(-1)
    
    return df.dropna().reset_index(drop=True)

# --- 2. LSTM MODEL ARCHITECTURE ---
class QuantLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 3. EXECUTION LOGIC ---
if ticker_input and train_button:
    df = get_advanced_data(ticker_input)
    latest = df.iloc[-1]

    # --- TOP METRIC ROW WITH TOOLTIPS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RSI (14D)", f"{latest['RSI']:.1f}", 
              help="**Relative Strength Index**: Measures momentum. Over 70 is 'Overbought' (sell signal), under 30 is 'Oversold' (buy signal).")
    m2.metric("MACD Hist", f"{latest['MACD_Hist']:.2f}", 
              help="**MACD Histogram**: Shows the trend strength. Rising green bars suggest bullish acceleration; falling red bars suggest bearish momentum.")
    m3.metric("BB Bandwidth", f"{(latest['BB_Upper'] - latest['BB_Lower']):.2f}", 
              help="**Bollinger Bandwidth**: Measures volatility. A narrow band (The Squeeze) often precedes a massive price explosion.")
    m4.metric("Fib 61.8% Level", f"${latest['Fib_618']:.2f}", 
              help="**The Golden Ratio**: A key support level. If the price stays above this during a pullback, the bull trend is likely to continue.")

    # --- AI TRAINING (LEAKAGE-FREE) ---
    features = ['Log_Ret', 'Vol_20', 'RSI', 'MACD_Hist']
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    
    scaler_X = StandardScaler().fit(train_df[features])
    scaler_y = StandardScaler().fit(train_df[['Target']])

    def prepare_seq(data, s_X, s_y):
        X_sc = s_X.transform(data[features])
        y_sc = s_y.transform(data[['Target']])
        xs, ys = [], []
        for i in range(len(X_sc) - seq_length):
            xs.append(X_sc[i:(i + seq_length)])
            ys.append(y_sc[i + seq_length])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    X_train, y_train = prepare_seq(train_df, scaler_X, scaler_y)
    model = QuantLSTM(len(features))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    with st.spinner("Training Neural Network..."):
        for _ in range(30):
            model.train()
            opt.zero_grad()
            loss = nn.MSELoss()(model(X_train), y_train)
            loss.backward()
            opt.step()

    # --- RECURSIVE FORWARD FORECAST ---
    forecast_days = int(n_years * 252)
    last_seq = scaler_X.transform(df[features].tail(seq_length))
    current_price = df['Close'].iloc[-1]
    preds = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(forecast_days):
            inp = torch.FloatTensor(last_seq).unsqueeze(0)
            ret_scaled = model(inp).item()
            ret = scaler_y.inverse_transform([[ret_scaled]])[0][0]
            
            # Stochastic Drift
            drifted_ret = ret + np.random.normal(0, df['Log_Ret'].std())
            current_price *= np.exp(drifted_ret)
            preds.append(current_price)
            
            # Update sequence for next day
            new_feat = [drifted_ret, df['Log_Ret'].std(), 50.0, 0.0]
            new_feat_sc = scaler_X.transform([new_feat])
            last_seq = np.append(last_seq[1:], new_feat_sc, axis=0)

    # --- ADVANCED PLOTLY DASHBOARD ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=('Price & Technicals', 'MACD', 'RSI'), row_width=[0.2, 0.2, 0.6])

    # Price, BB, and Forecast
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historic Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='rgba(173,216,230,0.5)', dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='rgba(173,216,230,0.5)', dash='dot'), fill='tonexty', name='BB Lower'), row=1, col=1)
    
    f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
    fig.add_trace(go.Scatter(x=f_dates, y=preds, name='AI Forecast', line=dict(color='red', width=3)), row=1, col=1)

    # MACD
    colors = ['green' if x > 0 else 'red' for x in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=900, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Enter a ticker and click 'Run Analysis' to generate the AI model.")
