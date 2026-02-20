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

# --- 1. INITIALIZATION & SIDEBAR ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="AI Quant Pro v16 - Risk Integrated", layout="wide")

if 'needs_retrain' not in st.session_state:
    st.session_state.needs_retrain = True

st.sidebar.header("🕹️ Control Panel")
ticker_input = st.sidebar.text_input("Ticker Symbol:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 3, value=1)
seq_length = st.sidebar.slider('Lookback Window (Days):', 10, 60, value=30)

if st.sidebar.button("🔄 Force Model Retrain"):
    st.session_state.needs_retrain = True

st.title(f"🧠 AI Quant & Risk Framework: {ticker_input}")

# --- 2. DATA & HYBRID MODEL ---
@st.cache_data
def get_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # Indicators
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / -df['Close'].diff().clip(upper=0).rolling(14).mean())))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD'].ewm(span=9).mean()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['Target'] = df['Log_Ret'].shift(-1)
    return df.dropna().reset_index(drop=True)

class HybridModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.cnn = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.gru = nn.GRU(32, 64, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x)).permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 3. EXECUTION ---
df = get_data(ticker_input)
features = ['Log_Ret', 'Vol_20', 'RSI', 'MACD_Hist']
split = int(len(df) * 0.8)
train_df = df.iloc[:split]

scaler_X = StandardScaler().fit(train_df[features])
scaler_y = StandardScaler().fit(train_df[['Target']])

if st.session_state.needs_retrain:
    X_sc = scaler_X.transform(train_df[features])
    y_sc = scaler_y.transform(train_df[['Target']])
    xs, ys = [], []
    for i in range(len(X_sc) - seq_length):
        xs.append(X_sc[i:i+seq_length]); ys.append(y_sc[i+seq_length])
    
    model = HybridModel(len(features))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(40):
        model.train(); opt.zero_grad()
        nn.MSELoss()(model(torch.FloatTensor(np.array(xs))), torch.FloatTensor(np.array(ys))).backward()
        opt.step()
    st.session_state.model = model
    st.session_state.needs_retrain = False

# --- 4. MULTI-PATH FORECAST & RISK ---
model = st.session_state.model
n_sims = 100
forecast_days = int(n_years * 252)
all_paths = []

model.eval()
with torch.no_grad():
    for _ in range(n_sims):
        last_seq = scaler_X.transform(df[features].tail(seq_length))
        current_price = df['Close'].iloc[-1]
        path = []
        for _ in range(forecast_days):
            inp = torch.FloatTensor(last_seq).unsqueeze(0)
            ret = scaler_y.inverse_transform([[model(inp).item()]])[0][0]
            # Add volatility shock
            ret += np.random.normal(0, df['Log_Ret'].std())
            current_price *= np.exp(ret)
            path.append(current_price)
            # Update seq
            new_feat = scaler_X.transform([[ret, df['Log_Ret'].std(), 50, 0]])
            last_seq = np.append(last_seq[1:], new_feat, axis=0)
        all_paths.append(path)

# Calculate Risk Metrics
final_prices = np.array([p[-1] for p in all_paths])
starting_price = df['Close'].iloc[-1]
total_returns = (final_prices - starting_price) / starting_price

var_95 = np.percentile(total_returns, 5)
cvar_95 = total_returns[total_returns <= var_95].mean()

# --- 5. RISK DASHBOARD ---
st.markdown("---")
st.subheader("🛡️ AI-Driven Risk Assessment")
r1, r2, r3 = st.columns(3)

r1.metric("95% Value at Risk (VaR)", f"{var_95*100:.1f}%", 
          help="**Value at Risk**: There is a 95% chance you will NOT lose more than this amount over the forecast period.")
r2.metric("Conditional VaR (CVaR)", f"{cvar_95*100:.1f}%", 
          help="**Expected Shortfall**: If the market enters the 'worst 5%' scenario, this is the average loss you can expect. It is more realistic than VaR.")
r3.metric("Max Forecasted Gain", f"{total_returns.max()*100:.1f}%", 
          help="The highest possible upside observed across all 100 AI simulations.")

with st.expander("📝 What do these numbers mean for me?"):
    st.write(f"""
    - **Conservative View:** You should be prepared for a potential drop of **{abs(var_95*100):.1f}%**. If you can't afford this, consider a smaller position.
    - **Worst Case (CVaR):** In a extreme market crash, the AI predicts an average drop of **{abs(cvar_95*100):.1f}%**.
    - **AI Verdict:** If the gain metric in the chart is significantly higher than the VaR, the 'Risk-Reward' ratio is mathematically favorable.
    """)

# --- 6. VISUALIZATION ---
fig = go.Figure()
for p in all_paths[:20]: # Show 20 sample paths
    fig.add_trace(go.Scatter(y=p, line=dict(width=0.5), opacity=0.2, showlegend=False))
fig.add_trace(go.Scatter(y=np.median(all_paths, axis=0), name="AI Median Forecast", line=dict(color='red', width=3)))
fig.update_layout(title="Monte Carlo Risk Paths", template="plotly_white", yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)
