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

# --- 1. MODEL ARCHITECTURE ---
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 2. HELPERS ---
def calculate_rsi(returns, window=14):
    gain = np.where(returns > 0, returns, 0)
    loss = np.where(returns < 0, -returns, 0)
    avg_gain = pd.Series(gain).rolling(window).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(window).mean().iloc[-1]
    return 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))

@st.cache_data
def load_and_process(symbol):
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['Target'] = df['Log_Ret'].shift(-1)
    return df.dropna().reset_index(drop=True)

def create_sequences(data, features, target_col, s_x, s_y, window):
    x_sc = s_x.transform(data[features])
    y_sc = s_y.transform(data[[target_col]])
    xs, ys = [], []
    for i in range(len(x_sc) - window):
        xs.append(x_sc[i:i+window])
        ys.append(y_sc[i+window])
    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

# --- 3. UI SETUP ---
st.sidebar.header("🕹️ Strategy Engine")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
lookback = st.sidebar.slider("Lookback Window:", 10, 60, 30)
epochs = st.sidebar.slider("Training Epochs:", 5, 100, 30)
n_splits = st.sidebar.slider("TSCV Folds:", 2, 5, 5)

df = load_and_process(ticker)

if df is not None:
    st.title(f"🚀 AI Quant: {ticker}")
    
    # Show Historical Chart Immediately
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price"))
    fig_hist.update_layout(template="plotly_dark", title="Historical Price Context", height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

    if st.sidebar.button("🔄 Run Full Analysis"):
        features = ['Log_Ret', 'Vol_20', 'RSI']
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        with st.status("Performing Time Series Cross-Validation...", expanded=True) as status:
            for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
                train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
                sc_x, sc_y = RobustScaler().fit(train_df[features]), RobustScaler().fit(train_df[['Target']])
                X_train, y_train = create_sequences(train_df, features, 'Target', sc_x, sc_y, lookback)
                X_test, y_test = create_sequences(test_df, features, 'Target', sc_x, sc_y, lookback)
                
                model = HybridQuantModel(len(features))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                for _ in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    nn.MSELoss()(model(X_train), y_train).backward()
                    optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    y_pred = sc_y.inverse_transform(model(X_test).numpy()).flatten()
                    y_actual = sc_y.inverse_transform(y_test.numpy()).flatten()
                    
                    strat_rets = np.where(y_pred > 0, 1, 0) * y_actual
                    sharpe = (np.mean(strat_rets) / (np.std(strat_rets) + 1e-9)) * np.sqrt(252)
                    fold_metrics.append(sharpe)
                    st.write(f"✅ Fold {i+1} Sharpe: {sharpe:.2f}")
            status.update(label="Validation Complete!", state="complete")

        # --- FEATURE IMPORTANCE (Permutation) ---
        st.subheader("🧠 Feature Importance")
        # Shuffle each feature and see how much MSE increases
        importances = []
        model.eval()
        with torch.no_grad():
            baseline_loss = nn.MSELoss()(model(X_test), y_test).item()
            for f_idx in range(len(features)):
                X_temp = X_test.clone()
                # Shuffle across the batch and time dimension for that specific feature
                X_temp[:, :, f_idx] = X_temp[torch.randperm(X_temp.size(0)), :, f_idx]
                shuffled_loss = nn.MSELoss()(model(X_temp), y_test).item()
                importances.append(max(0, shuffled_loss - baseline_loss))
        
        imp_df = pd.DataFrame({'Feature': features, 'Impact': importances}).sort_values(by='Impact', ascending=False)
        st.bar_chart(imp_df.set_index('Feature'))

        # --- FORECAST ---
        st.subheader("🔮 Recursive Forecast (Next 15 Days)")
        sc_x_f, sc_y_f = RobustScaler().fit(df[features]), RobustScaler().fit(df[['Target']])
        X_f, y_f = create_sequences(df, features, 'Target', sc_x_f, sc_y_f, lookback)
        
        # Train production model
        prod_model = HybridQuantModel(len(features))
        opt_f = torch.optim.Adam(prod_model.parameters(), lr=0.001)
        for _ in range(epochs):
            prod_model.train()
            opt_f.zero_grad()
            nn.MSELoss()(prod_model(X_f), y_f).backward()
            opt_f.step()

        prod_model.eval()
        current_win = df[features].tail(lookback).values
        prices = [df['Close'].iloc[-1]]
        
        for _ in range(15):
            win_t = torch.FloatTensor(sc_x_f.transform(current_win)).unsqueeze(0)
            with torch.no_grad():
                p_ret = sc_y_f.inverse_transform([[prod_model(win_t).item()]])[0][0]
            
            prices.append(prices[-1] * np.exp(p_ret))
            new_row = np.array([p_ret, np.std(np.append(current_win[:, 0], p_ret)[-20:]), calculate_rsi(np.append(current_win[:, 0], p_ret))])
            current_win = np.append(current_win[1:], [new_row], axis=0)

        f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(16)]
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=f_dates, y=prices, line=dict(color='#00ffcc', width=4)))
        fig_f.update_layout(template="plotly_dark", title="Predicted Price Path")
        st.plotly_chart(fig_f, use_container_width=True)
else:
    st.error("Ticker not found. Please check the symbol.")
