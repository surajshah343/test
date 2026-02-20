import streamlit as st
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as go
from plotly.subplots import make_subplots

# --- 1. INITIALIZATION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="AI Quant Pro v16 - Hybrid CNN-GRU", layout="wide")

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
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    # Feature Engineering (Strictly Vectorized)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Target: Next day's Log Return
    df['Target'] = df['Log_Ret'].shift(-1)
    return df.dropna().reset_index(drop=True)

# --- 4. HYBRID CNN-GRU ARCHITECTURE ---
# CNN extracts local patterns (e.g., "head and shoulders" shapes)
# GRU extracts long-term temporal dependencies
class HybridQuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gru = nn.GRU(64, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: [Batch, Seq, Features] -> Conv1d needs [Batch, Features, Seq]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # Back to [Batch, Seq, Features]
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]) # Predict based on last hidden state

# --- 5. EXECUTION PIPELINE ---
df = load_and_process(ticker)
features = ['Log_Ret', 'Vol_20', 'RSI']

# Splitting Data (80% Train, 20% Backtest)
split = int(len(df) * 0.8)
train_df = df.iloc[:split].copy()
test_df = df.iloc[split:].copy()

# Scaling (Only fit on Training to prevent leakage!)
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    progress = st.progress(0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        progress.progress((epoch + 1) / epochs)
    
    st.session_state.model = model
    st.session_state.needs_retrain = False
    st.success("Model Training Complete.")

# --- 6. BACKTESTING VALIDATION ---
model = st.session_state.model
X_test, y_test = create_sequences(test_df, scaler_X, scaler_y, lookback)

model.eval()
with torch.no_grad():
    y_pred_sc = model(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_sc.numpy())
    y_actual = scaler_y.inverse_transform(y_test.numpy())

# Calculate directional accuracy
correct_dir = np.sum(np.sign(y_pred) == np.sign(y_actual)) / len(y_actual)

# --- 7. FUTURE FORECASTING (Recursive) ---
last_window = df[features].tail(lookback).values
forecast_preds = []
current_price = df['Close'].iloc[-1]
price_path = [current_price]

with torch.no_grad():
    temp_window = last_window.copy()
    for _ in range(forecast_horizon):
        # Scale window
        win_sc = scaler_X.transform(temp_window)
        win_tensor = torch.FloatTensor(win_sc).unsqueeze(0)
        
        # Predict next log return
        pred_ret_sc = model(win_tensor).item()
        pred_ret = scaler_y.inverse_transform([[pred_ret_sc]])[0][0]
        
        # Update price and feature window
        new_price = price_path[-1] * np.exp(pred_ret)
        price_path.append(new_price)
        
        # Simulated Feature Update
        new_row = np.array([[pred_ret, temp_window[:, 1].mean(), temp_window[:, 2].mean()]])
        temp_window = np.append(temp_window[1:], new_row, axis=0)

# --- 8. DASHBOARD VISUALIZATION ---
st.title(f"📊 Quant Diagnostics: {ticker}")

col1, col2, col3 = st.columns(3)
col1.metric("Backtest Accuracy", f"{correct_dir:.1%}", help="How often the model correctly predicted if the price would go Up or Down.")
col2.metric("Predicted 20-Day Move", f"{((price_path[-1]/price_path[0])-1):.2%}")
col3.metric("Last RSI", f"{df['RSI'].iloc[-1]:.1f}")

# Main Chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

# Actual vs Predicted (Backtest)
test_dates = df['Date'].iloc[split+lookback:]
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical"), row=1, col=1)

# Forecast
f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_horizon+1)]
fig.add_trace(go.Scatter(x=f_dates, y=price_path[1:], name="AI Forecast", line=dict(color='red', dash='dot')), row=1, col=1)

# Indicators
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='gray')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=800, template="plotly_dark", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.info("**Note:** Directional Accuracy above 55% is considered exceptional in quantitative finance. If your accuracy is 50%, the model is essentially flipping a coin.")
