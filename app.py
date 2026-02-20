import streamlit as st
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as go
from plotly.subplots import make_subplots

# --- 1. INITIALIZATION & SIDEBAR (Prevents NameError) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="AI Quant Pro v15 - Hybrid CNN-GRU", layout="wide")

# Initialize session state for retraining logic
if 'needs_retrain' not in st.session_state:
    st.session_state.needs_retrain = True

st.sidebar.header("🕹️ Control Panel")
ticker_input = st.sidebar.text_input("Ticker Symbol:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 3, value=1)
seq_length = st.sidebar.slider('Lookback Window (Days):', 10, 60, value=30)

# The Retrain Button
if st.sidebar.button("🔄 Force Model Retrain"):
    st.session_state.needs_retrain = True

st.title(f"🧠 Hybrid CNN-GRU Quant Framework: {ticker_input}")

# --- 2. DATA ENGINE (Technicals & Math) ---
@st.cache_data
def get_advanced_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)

    # Fibonacci (52-week)
    recent = df.tail(252)
    h, l = recent['High'].max(), recent['Low'].min()
    df['Fib_618'] = h - 0.382 * (h - l)

    # Stationary Target
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['Target'] = df['Log_Ret'].shift(-1)
    
    return df.dropna().reset_index(drop=True)

# --- 3. HYBRID CNN-GRU ARCHITECTURE ---
class HybridModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # CNN extracts spatial patterns from the lookback window
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        # GRU handles temporal memory
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch, seq, features] -> CNN needs [batch, features, seq]
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))
        x = x.permute(0, 2, 1) # Back to [batch, seq, features]
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 4. EXECUTION FLOW ---
df = get_advanced_data(ticker_input)
latest = df.iloc[-1]

# Interactive Metrics with Tooltips
m1, m2, m3, m4 = st.columns(4)
m1.metric("RSI", f"{latest['RSI']:.1f}", help="Standard RSI. >70 is overbought, <30 is oversold.")
m2.metric("MACD Hist", f"{latest['MACD_Hist']:.2f}", help="Difference between MACD and Signal Line. Rising = Stronger trend.")
m3.metric("BB Bandwidth", f"{(latest['BB_Upper'] - latest['BB_Lower']):.2f}", help="Market volatility. Narrow = potential breakout.")
m4.metric("Fib 61.8%", f"${latest['Fib_618']:.2f}", help="Major support/resistance level. Often called the 'Golden Ratio'.")

# --- 5. DEEP LEARNING WORKFLOW ---
features = ['Log_Ret', 'Vol_20', 'RSI', 'MACD_Hist']
split = int(len(df) * 0.8)
train_df = df.iloc[:split]

scaler_X = StandardScaler().fit(train_df[features])
scaler_y = StandardScaler().fit(train_df[['Target']])

def get_sequences(data, s_X, s_y):
    X_sc = s_X.transform(data[features])
    y_sc = s_y.transform(data[['Target']])
    xs, ys = [], []
    for i in range(len(X_sc) - seq_length):
        xs.append(X_sc[i:(i + seq_length)])
        ys.append(y_sc[i + seq_length])
    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

if st.session_state.needs_retrain:
    X_train, y_train = get_sequences(train_df, scaler_X, scaler_y)
    model = HybridModel(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch + 1) / 50)
        status_text.text(f"Training Hybrid CNN-GRU... Loss: {loss.item():.5f}")
    
    st.session_state.model = model
    st.session_state.needs_retrain = False
    status_text.success("Model Trained Successfully!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

# --- 6. FORECASTING & VIZ ---
model = st.session_state.model
forecast_days = int(n_years * 252)
last_seq = scaler_X.transform(df[features].tail(seq_length))
price_trace = [df['Close'].iloc[-1]]

model.eval()
with torch.no_grad():
    for _ in range(forecast_days):
        inp = torch.FloatTensor(last_seq).unsqueeze(0)
        ret_sc = model(inp).item()
        ret = scaler_y.inverse_transform([[ret_sc]])[0][0]
        
        # Adding stochastic volatility for realism
        real_ret = ret + np.random.normal(0, df['Log_Ret'].std() * 0.5)
        price_trace.append(price_trace[-1] * np.exp(real_ret))
        
        # Recursive Sequence Update
        new_row = [real_ret, df['Log_Ret'].std(), 50.0, 0.0] 
        new_sc = scaler_X.transform([new_row])
        last_seq = np.append(last_seq[1:], new_sc, axis=0)

# Final Visualization
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historic'), row=1, col=1)
f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
fig.add_trace(go.Scatter(x=f_dates, y=price_trace[1:], name='AI Forecast', line=dict(color='red', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)

fig.update_layout(height=700, template="plotly_white", hovermode="x unified", title=f"Hybrid Deep Learning Forecast: {ticker_input}")
st.plotly_chart(fig, use_container_width=True)
