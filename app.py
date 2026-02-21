import streamlit as st
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from plotly import graph_objs as go

# --- 1. INITIALIZATION & STYLING ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="AI Quant Pro v18.0", layout="wide")

# --- 2. MODEL ARCHITECTURE ---
class HybridQuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64)
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 3. CORE UTILITIES ---
def calculate_rsi(returns, window=14):
    delta = returns
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(window).mean().iloc[-1]
    if avg_loss == 0: return 100
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data
def load_and_process(symbol):
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        
        # Features
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

def create_sequences(data, features, target_col, s_x, s_y, window):
    x_sc = s_x.transform(data[features])
    y_sc = s_y.transform(data[[target_col]])
    xs, ys = [], []
    for i in range(len(x_sc) - window):
        xs.append(x_sc[i:i+window])
        ys.append(y_sc[i+window])
    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

# --- 4. STRATEGY ENGINE ---
st.sidebar.header("🕹️ Strategy Engine")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
lookback = st.sidebar.slider("Lookback Window:", 10, 60, 30)
epochs = st.sidebar.slider("Training Epochs:", 5, 100, 30)
n_splits = st.sidebar.slider("TSCV Folds:", 2, 5, 3)

df = load_and_process(ticker)

if df is not None:
    features = ['Log_Ret', 'Vol_20', 'RSI']
    st.title(f"🚀 AI Quant: {ticker}")
    
    if st.sidebar.button("🔄 Run TSCV Backtest"):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_acc, fold_sharpe = [], []

        for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            
            # Scalers (Robust to handle fat-tails in returns)
            sc_x, sc_y = RobustScaler().fit(train_df[features]), RobustScaler().fit(train_df[['Target']])
            
            X_train, y_train = create_sequences(train_df, features, 'Target', sc_x, sc_y, lookback)
            X_test, y_test = create_sequences(test_df, features, 'Target', sc_x, sc_y, lookback)
            
            model = HybridQuantModel(len(features))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training Loop
            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(X_train)
                loss = nn.MSELoss()(pred, y_train)
                loss.backward()
                optimizer.step()
            
            # Backtest Fold
            model.eval()
            with torch.no_grad():
                y_pred_sc = model(X_test).numpy()
                y_pred = sc_y.inverse_transform(y_pred_sc).flatten()
                y_actual = sc_y.inverse_transform(y_test.numpy()).flatten()
                
                # Signal: 1 if predicted return > 0 else 0
                signals = np.where(y_pred > 0, 1, 0)
                strategy_returns = signals * y_actual
                
                # Metrics
                acc = np.mean(np.sign(y_pred) == np.sign(y_actual))
                sharpe = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252)
                
                fold_acc.append(acc)
                fold_sharpe.append(sharpe)
                st.write(f"Fold {i+1} | Acc: {acc:.2%} | Sharpe: {sharpe:.2f}")

        # Summary Metrics
        c1, c2 = st.columns(2)
        c1.metric("Avg Directional Accuracy", f"{np.mean(fold_acc):.2%}")
        c2.metric("Avg Sharpe Ratio", f"{np.mean(fold_sharpe):.2f}")

        # --- FINAL FORECAST ---
        st.subheader("🔮 15-Day Recursive Projection")
        sc_x_f, sc_y_f = RobustScaler().fit(df[features]), RobustScaler().fit(df[['Target']])
        X_f, y_f = create_sequences(df, features, 'Target', sc_x_f, sc_y_f, lookback)
        
        prod_model = HybridQuantModel(len(features))
        opt_f = torch.optim.Adam(prod_model.parameters(), lr=0.001)
        for _ in range(epochs):
            prod_model.train()
            opt_f.zero_grad()
            nn.MSELoss()(prod_model(X_f), y_f).backward()
            opt_f.step()

        prod_model.eval()
        last_win = df[features].tail(lookback).values
        forecast_rets = []
        current_win = last_win.copy()

        for _ in range(15):
            win_sc = sc_x_f.transform(current_win)
            win_t = torch.FloatTensor(win_sc).unsqueeze(0)
            with torch.no_grad():
                p_sc = prod_model(win_t).item()
                p_ret = sc_y_f.inverse_transform([[p_sc]])[0][0]
            
            forecast_rets.append(p_ret)
            
            # RECURSIVE FEATURE UPDATE
            next_returns_stream = np.append(current_win[:, 0], p_ret)
            new_vol = np.std(next_returns_stream[-20:])
            new_rsi = calculate_rsi(next_returns_stream)
            new_row = np.array([p_ret, new_vol, new_rsi])
            current_win = np.append(current_win[1:], [new_row], axis=0)

        # Plotting
        prices = [df['Close'].iloc[-1]]
        for r in forecast_rets: prices.append(prices[-1] * np.exp(r))
        
        f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(16)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'].tail(100), y=df['Close'].tail(100), name="Historical"))
        fig.add_trace(go.Scatter(x=f_dates, y=prices, name="AI Forecast", line=dict(color='orange', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter a ticker and run the backtest.")
