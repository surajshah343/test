import streamlit as st
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from plotly import graph_objs as go

# --- 1. INITIALIZATION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="AI Quant Pro v17.0 - TSCV", layout="wide")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Strategy Engine")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
lookback = st.sidebar.slider("Lookback Window (Days):", 10, 60, 30)
epochs = st.sidebar.slider("Training Epochs:", 5, 50, 20)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days):", 5, 30, 15)
n_splits = st.sidebar.slider("Cross-Val Folds:", 2, 5, 3)

# --- 3. DATA ARCHITECTURE ---
@st.cache_data
def load_and_process(symbol):
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        
        # Feature Engineering (Standard technicals)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        df['Target'] = df['Log_Ret'].shift(-1)
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

# --- 4. MODEL ARCHITECTURE ---
class HybridQuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2) # [Batch, Features, Seq]
        x = self.cnn(x)
        x = x.transpose(1, 2) # [Batch, Seq, 32]
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, features, target_col, s_x, s_y, window):
    x_sc = s_x.transform(data[features])
    y_sc = s_y.transform(data[[target_col]])
    xs, ys = [], []
    for i in range(len(x_sc) - window):
        xs.append(x_sc[i:i+window])
        ys.append(y_sc[i+window])
    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

# --- 5. EXECUTION WITH TSCV ---
df = load_and_process(ticker)

if df is not None:
    features = ['Log_Ret', 'Vol_20', 'RSI']
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    st.title(f"🚀 Quant Hub: {ticker} (TSCV Mode)")
    
    # Progress and Metrics containers
    fold_metrics = []
    
    if st.sidebar.button("🔄 Run TSCV Backtest"):
        # Image showing how TimeSeriesSplit works: training on expanding windows
        # 
        
        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            st.write(f"📊 Processing Fold {i+1}...")
            
            train_df = df.iloc[train_index].copy()
            test_df = df.iloc[test_index].copy()
            
            # Local scalers for each fold to prevent leakage
            scaler_X = StandardScaler().fit(train_df[features])
            scaler_y = StandardScaler().fit(train_df[['Target']])
            
            X_train, y_train = create_sequences(train_df, features, 'Target', scaler_X, scaler_y, lookback)
            X_test, y_test = create_sequences(test_df, features, 'Target', scaler_X, scaler_y, lookback)
            
            # Model Training
            model = HybridQuantModel(len(features))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward()
                optimizer.step()
            
            # Fold Validation
            model.eval()
            with torch.no_grad():
                y_pred_sc = model(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_sc.numpy())
                y_actual = scaler_y.inverse_transform(y_test.numpy())
                
                # Metric: Directional Accuracy
                acc = np.sum(np.sign(y_pred) == np.sign(y_actual)) / len(y_actual)
                fold_metrics.append(acc)
                st.write(f"Fold {i+1} Directional Accuracy: {acc:.2%}")

        avg_acc = np.mean(fold_metrics)
        st.metric("Aggregate TSCV Accuracy", f"{avg_acc:.2%}")

        # --- FINAL PRODUCTION TRAINING ---
        # Train on the most recent full dataset for the actual forecast
        scaler_X_final = StandardScaler().fit(df[features])
        scaler_y_final = StandardScaler().fit(df[['Target']])
        X_final, y_final = create_sequences(df, features, 'Target', scaler_X_final, scaler_y_final, lookback)
        
        prod_model = HybridQuantModel(len(features))
        optimizer_prod = torch.optim.Adam(prod_model.parameters(), lr=0.001)
        
        for _ in range(epochs):
            prod_model.train()
            optimizer_prod.zero_grad()
            loss = nn.MSELoss()(prod_model(X_final), y_final)
            loss.backward()
            optimizer_prod.step()

        # --- 6. FORECAST ---
        prod_model.eval()
        with torch.no_grad():
            last_win_raw = df[features].tail(lookback).values
            price_path = [df['Close'].iloc[-1]]
            curr_win_raw = last_win_raw.copy()
            
            for _ in range(forecast_horizon):
                win_sc = scaler_X_final.transform(curr_win_raw)
                win_tensor = torch.FloatTensor(win_sc).unsqueeze(0)
                p_ret_sc = prod_model(win_tensor).item()
                p_ret = scaler_y_final.inverse_transform([[p_ret_sc]])[0][0]
                
                price_path.append(price_path[-1] * np.exp(p_ret))
                
                # Recursive update keeping Vol/RSI constant for short-term projection
                new_row = np.array([[p_ret, curr_win_raw[-1, 1], curr_win_raw[-1, 2]]])
                curr_win_raw = np.append(curr_win_raw[1:], new_row, axis=0)

        # --- 7. VISUALIZATION ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Price"))
        
        last_date = df['Date'].iloc[-1]
        f_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
        
        fig.add_trace(go.Scatter(
            x=f_dates, 
            y=price_path[1:], 
            name="TSCV-Validated Forecast",
            line=dict(color='orange', width=3, dash='dot')
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            title=f"AI Forecast for {ticker} (Horizontal Line = Random Walk baseline)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Run TSCV Backtest' in the sidebar to begin statistical validation.")
else:
    st.error("Invalid Ticker or No Data Found.")
