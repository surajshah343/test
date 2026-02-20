import streamlit as st
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time  # Fix for the previous NameError
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as go
from plotly.subplots import make_subplots

# --- 1. INITIALIZATION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="AI Quant Pro v16.1", layout="wide")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'needs_retrain' not in st.session_state:
    st.session_state.needs_retrain = True

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Strategy Engine")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
lookback = st.sidebar.slider("Lookback Window (Days):", 10, 60, 30)
epochs = st.sidebar.slider("Training Epochs:", 10, 100, 50)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days):", 5, 60, 20)

if st.sidebar.button("🔄 Train & Validate Model"):
    st.session_state.needs_retrain = True

# --- 3. DATA ARCHITECTURE ---
@st.cache_data
def load_and_process(symbol):
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        
        # Feature Engineering
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df['Target'] = df['Log_Ret'].shift(-1)
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

# --- 4. MODEL ARCHITECTURE ---
class HybridQuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input_dim is the number of features (3)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x input shape: [Batch, Seq_Len, Features]
        # Conv1d requires: [Batch, Features, Seq_Len]
        x = x.transpose(1, 2) 
        x = self.cnn(x)
        x = x.transpose(1, 2) # Back to [Batch, Seq_Len, 64]
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 5. EXECUTION ---
df = load_and_process(ticker)

if df is not None:
    features = ['Log_Ret', 'Vol_20', 'RSI']
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    scaler_X = StandardScaler().fit(train_df[features])
    scaler_y = StandardScaler().fit(train_df[['Target']])

    def create_sequences(data, s_x, s_y, window):
        x_sc = s_x.transform(data[features])
        y_sc = s_y.transform(data[['Target']])
        xs, ys = [], []
        for i in range(len(x_sc) - window):
            xs.append(x_sc[i:i+window])
            ys.append(y_sc[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    if st.session_state.needs_retrain:
        X_train, y_train = create_sequences(train_df, scaler_X, scaler_y, lookback)
        model = HybridQuantModel(len(features))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        prog_bar = st.progress(0)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X_train), y_train)
            loss.backward()
            optimizer.step()
            prog_bar.progress((epoch+1)/epochs)
        
        st.session_state.model = model
        st.session_state.needs_retrain = False

    # --- 6. VALIDATION & FORECAST ---
    model = st.session_state.model
    X_test, y_test = create_sequences(test_df, scaler_X, scaler_y, lookback)

    model.eval()
    with torch.no_grad():
        y_pred_sc = model(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_sc.numpy())
        y_actual = scaler_y.inverse_transform(y_test.numpy())
        
        # Recursive Future Forecast
        last_win = df[features].tail(lookback).values
        price_path = [df['Close'].iloc[-1]]
        curr_win = last_win.copy()
        
        for _ in range(forecast_horizon):
            win_sc = scaler_X.transform(curr_win)
            win_tensor = torch.FloatTensor(win_sc).unsqueeze(0)
            p_ret_sc = model(win_tensor).item()
            p_ret = scaler_y.inverse_transform([[p_ret_sc]])[0][0]
            
            price_path.append(price_path[-1] * np.exp(p_ret))
            # Mock update for RSI/Vol features for recursion
            new_row = np.array([[p_ret, curr_win[:,1].mean(), 50.0]])
            curr_win = np.append(curr_win[1:], new_row, axis=0)

    # --- 7. VIZ ---
    st.title(f"🚀 Quant Hub: {ticker}")
    acc = np.sum(np.sign(y_pred) == np.sign(y_actual)) / len(y_actual)
    
    c1, c2 = st.columns([1, 3])
    c1.metric("Directional Accuracy", f"{acc:.1%}")
    c1.write("The model correctly predicted the 'Up/Down' movement this often in testing.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="History"))
    f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_horizon+1)]
    fig.add_trace(go.Scatter(x=f_dates, y=price_path[1:], name="AI Forecast", line=dict(color='orange', dash='dot')))
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Could not fetch data. Please check the ticker symbol.")
