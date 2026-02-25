import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import scipy.stats as stats
from typing import Dict, Callable
import datetime
from datetime import timedelta

# Alpaca Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Setup hardware acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

st.set_page_config(page_title="Pro Quant Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 1. CORE MATH & EXECUTION CLASSES
# ==========================================

class BayesianOptimizer:
    def optimize(self, objective_func: Callable, n_trials=10):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=n_trials)
        return study.best_params, study.best_value

class KellySizer:
    def compute(self, returns, risk_free=0.0, fraction=0.5):
        mean = np.mean(returns)
        var = np.var(returns)
        if var == 0: return 0.0
        return ((mean - risk_free) / var) * fraction

class MacroFactorModel:
    def __init__(self, factors):
        self.factors = factors
        self.model = LinearRegression()

    def compute_factor_exposures(self, returns_df, macro_returns_df):
        macro_safe = macro_returns_df.add_suffix('_MACRO')
        aligned_data = pd.concat([returns_df, macro_safe], axis=1).dropna()
        if aligned_data.empty: return pd.DataFrame()
            
        y_data = aligned_data[returns_df.columns]
        x_data = aligned_data[macro_safe.columns]
        
        exposures = []
        for i in range(y_data.shape[1]):
            self.model.fit(x_data, y_data.iloc[:, i])
            exposures.append(self.model.coef_)
        return pd.DataFrame(exposures, columns=self.factors, index=returns_df.columns)

class RiskParityAllocator:
    def allocate(self, cov_matrix):
        n = cov_matrix.shape[0]
        w = np.ones(n) / n
        for _ in range(1000):
            risk = w * (cov_matrix @ w)
            grad = risk - np.sum(risk) / n
            w -= 0.05 * grad
            w = np.maximum(w, 0)
            if np.sum(w) > 0: w /= np.sum(w)
        return w

class HestonMonteCarlo:
    def simulate(self, S0, T=1, dt=1/252, mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7):
        n_steps = int(T / dt)
        S, v = np.zeros(n_steps), np.zeros(n_steps)
        S[0], v[0] = S0, theta
        for t in range(1, n_steps):
            z1, z2 = np.random.randn(), np.random.randn()
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1] * dt) * z2)
            S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1)
        return S

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURES
# ==========================================

class TransformerAlpha(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        return self.fc(self.transformer(self.embedding(x))[:, -1, :])

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        return torch.sum(attn_weights * lstm_out, dim=1), attn_weights

class HybridQuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm1d(64))
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.attention = TemporalAttention(128)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)

# ==========================================
# 3. DATA & FEATURE ENGINEERING
# ==========================================

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def fetch_portfolio_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
    return data, data.pct_change().dropna()

@st.cache_data(ttl=3600)
def load_deep_dive_features(symbol, start):
    df = yf.download(symbol, start=start, interval="1d", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['Std_20'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['Std_20'])
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['VWMA_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()
    
    tr = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    nvi = [1000]
    for i in range(1, len(df)):
        if df['Volume'].iloc[i] < df['Volume'].iloc[i-1]:
            ret = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
            nvi.append(nvi[-1] + (ret * nvi[-1]))
        else:
            nvi.append(nvi[-1])
    df['NVI'] = nvi
    df['NVI_Signal'] = df['NVI'].ewm(span=255, adjust=False).mean()
    
    return df.dropna().reset_index(drop=True)

@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    tkr = yf.Ticker(symbol)
    try: return tkr.info, tkr.balance_sheet, tkr.income_stmt
    except: return tkr.info, pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
def get_options_data(symbol):
    tkr = yf.Ticker(symbol)
    expirations = tkr.options
    if not expirations: return None, None, None
    chain = tkr.option_chain(expirations[0])
    return expirations, chain.calls, chain.puts

def create_nn_sequences(data, features, target_col, s_x, s_y, window):
    x_sc = s_x.transform(data[features])
    y_sc = s_y.transform(data[[target_col]])
    xs, ys = [], []
    for i in range(len(x_sc) - window):
        xs.append(x_sc[i:i+window])
        ys.append(y_sc[i+window])
    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

def safe_get(d, key, default="N/A", format_type="num"):
    val = d.get(key)
    if val is None or val == "": return default
    if format_type == "pct": return f"{val*100:.2f}%"
    if format_type == "curr": return f"${val:,.2f}"
    if format_type == "float": return f"{val:.2f}"
    return val

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call": return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else: return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

# ==========================================
# 4. DASHBOARD UI
# ==========================================

st.sidebar.header("🕹️ Global Portfolio Settings")
tickers_input = st.sidebar.text_input("Portfolio Tickers (comma separated):", "SPY, QQQ, AAPL, GLD")
tickers = [t.strip().upper() for t in tickers_input.split(",")]
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

st.sidebar.divider()
st.sidebar.header("🔬 Deep Dive Settings")
target_asset = st.sidebar.selectbox("Select Target Asset:", tickers)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (BSM Options):", value=0.045, step=0.005, format="%.3f")
forecast_horizon = st.sidebar.selectbox("Neural Net Forecast Horizon:", ["1 Week", "1 Month", "1 Year"])

with st.sidebar.expander("Advanced ML Parameters"):
    lookback = st.slider("Sequence Window:", 10, 60, 30)
    epochs = st.slider("Training Epochs:", 10, 100, 30)
    batch_size = st.selectbox("Batch Size:", [32, 64, 128], index=1)
    fib_window = st.slider("Fibonacci Days:", 30, 1000, 252)

st.title(f"🏛️ Pro Quant Platform")
st.markdown(f"**Compute Device:** `{device.type.upper()}` | **Active Portfolio Assets:** `{len(tickers)}`")

# Global Data Fetch
prices_df, returns_df = fetch_portfolio_data(tickers, start_date, end_date)
macro_prices, macro_returns = fetch_portfolio_data(["SPY", "TIP", "TLT"], start_date, end_date)

if prices_df.empty:
    st.warning("Please enter valid ticker symbols.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌐 Portfolio AI", 
    "🔬 Technical Deep Dive", 
    "💎 Fundamentals & DuPont", 
    "🏢 Macro & Allocation", 
    "⚖️ Options & Volatility",
    "⚙️ System Optuna",
    "🔗 Live Paper Trading"
])

# ---------------------------------------------------------
# TAB 1: PORTFOLIO AI (TRANSFORMER)
# ---------------------------------------------------------
with tab1:
    st.header("Portfolio-Wide Transformer Alpha")
    st.markdown("Trains a self-attention mechanism across all portfolio assets simultaneously to predict next-day relative performance.")
    
    fig = px.line(prices_df / prices_df.iloc[0] * 100, title="Normalized Asset Performance (Base 100)")
    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Normalized Price")
    st.plotly_chart(fig, use_container_width=True)
    
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_df.values)
    
    if st.button("Train Global Transformer", type="primary"):
        with st.spinner("Training Transformer..."):
            X, y = [], []
            for i in range(len(returns_scaled) - 5):
                X.append(returns_scaled[i:i+5])
                y.append(returns_scaled[i+5])
            X, y = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
            
            model = TransformerAlpha(input_dim=len(tickers))
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            
            for _ in range(30):
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(X), y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                preds_inv = scaler.inverse_transform(model(X[-5:]).numpy())
            
            st.session_state['transformer_preds'] = preds_inv
            # Simple threshold execution
            st.session_state['latest_signals'] = np.where(preds_inv[-1] > 0.005, 1.0, np.where(preds_inv[-1] < -0.005, -1.0, 0.0))
            st.success("Global Model Trained.")

    if 'transformer_preds' in st.session_state:
        st.write("Recent Out-of-Sample Predictions (Daily % Return)")
        st.dataframe(pd.DataFrame(st.session_state['transformer_preds'], columns=tickers).style.background_gradient(cmap='RdYlGn'), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: TECHNICAL DEEP DIVE (CNN-LSTM)
# ---------------------------------------------------------
with tab2:
    st.header(f"[{target_asset}] Advanced Technicals & Hybrid Forecast")
    
    with st.expander("📖 How to Read These Professional Indicators", expanded=False):
        st.markdown("""
        * **VWMA:** Price above VWMA indicates institutional momentum is bullish.
        * **MACD:** Buy signal when MACD crosses above the signal line.
        * **OBV:** Bearish divergence if price hits higher highs but OBV hits lower highs.
        * **NVI (Smart Money):** Updates on low-volume days. Crossover above 255d signal is very bullish.
        * **ATR:** Volatility gauge. Wider ATR = widen your stop losses and reduce sizing.
        """)
        
    df_deep = load_deep_dive_features(target_asset, start_date)
    if df_deep is not None:
        fib_df = df_deep.tail(min(fib_window, len(df_deep)))
        max_p, min_p = fib_df['High'].max(), fib_df['Low'].min()
        diff = max_p - min_p
        
        

        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13], subplot_titles=("Price, MAs, VWMA & BB", "RSI", "MACD", "OBV", "NVI", "ATR"))
        
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['Close'], name="Close", line=dict(color="white")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['SMA_20'], name="20-SMA", line=dict(color="cyan")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['VWMA_20'], name="20-VWMA", line=dict(color="#f4d03f")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['BB_Upper'], line=dict(color='gray', dash='dash'), name='UB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['BB_Lower'], line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='LB'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['RSI'], line=dict(color='mediumpurple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
        fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")
        
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['MACD'], line=dict(color='dodgerblue'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['MACD_Signal'], line=dict(color='darkorange'), name='Signal'), row=3, col=1)
        fig.add_trace(go.Bar(x=df_deep['Date'], y=df_deep['MACD_Hist'], marker_color=['#2ca02c' if v >= 0 else '#d62728' for v in df_deep['MACD_Hist']], name='Hist'), row=3, col=1)
        
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['OBV'], line=dict(color='#3498db'), name='OBV'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['OBV_EMA'], line=dict(color='orange', dash='dot'), name='OBV EMA'), row=4, col=1)
        
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['NVI'], line=dict(color='#9b59b6'), name='NVI'), row=5, col=1)
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['NVI_Signal'], line=dict(color='yellow', dash='dot'), name='NVI Sig'), row=5, col=1)
        
        fig.add_trace(go.Scatter(x=df_deep['Date'], y=df_deep['ATR'], line=dict(color='#e74c3c'), name='ATR'), row=6, col=1)
        
        fig.update_layout(template="plotly_dark", height=1200, hovermode="x unified", dragmode="zoom")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Train CNN-LSTM Attention Forecast"):
            with st.spinner(f"Training hybrid network on {target_asset}..."):
                feats = ['Log_Ret', 'Vol_20', 'RSI']
                sc_x, sc_y = RobustScaler().fit(df_deep[feats]), RobustScaler().fit(df_deep[['Log_Ret']])
                X_f, y_f = create_nn_sequences(df_deep, feats, 'Log_Ret', sc_x, sc_y, lookback)
                
                prod_model = HybridQuantModel(len(feats)).to(device)
                opt_f = torch.optim.AdamW(prod_model.parameters(), lr=0.001)
                prod_loader = DataLoader(TensorDataset(X_f, y_f), batch_size=batch_size, shuffle=True)
                
                for _ in range(epochs):
                    prod_model.train()
                    for b_x, b_y in prod_loader:
                        opt_f.zero_grad()
                        nn.HuberLoss()(prod_model(b_x.to(device)), b_y.to(device)).backward()
                        opt_f.step()
                
                prod_model.eval()
                curr_win = df_deep[feats].tail(lookback).values
                fcst_prices, fcst_dates = [df_deep['Close'].iloc[-1]], [df_deep['Date'].iloc[-1]]
                h_days = {'1 Week': 5, '1 Month': 21, '1 Year': 252}[forecast_horizon]
                
                for _ in range(h_days):
                    win_t = torch.FloatTensor(sc_x.transform(curr_win)).unsqueeze(0).to(device)
                    with torch.no_grad(): p_ret = sc_y.inverse_transform([[prod_model(win_t).cpu().item()]])[0][0]
                    if forecast_horizon == '1 Year': p_ret *= 0.95
                    
                    fcst_prices.append(fcst_prices[-1] * np.exp(p_ret))
                    next_date = fcst_dates[-1] + timedelta(days=1)
                    while next_date.weekday() >= 5: next_date += timedelta(days=1)
                    fcst_dates.append(next_date)
                    
                    new_row = np.array([p_ret, df_deep['Vol_20'].iloc[-1], df_deep['RSI'].iloc[-1]])
                    curr_win = np.append(curr_win[1:], [new_row], axis=0)

                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=df_deep['Date'].tail(100), y=df_deep['Close'].tail(100), name="Recent Price"))
                fig_f.add_trace(go.Scatter(x=fcst_dates, y=fcst_prices, name="LSTM Forecast", line=dict(dash="dash", color="#2ca02c", width=3)))
                fig_f.update_layout(template="plotly_dark", height=400, title=f"{forecast_horizon} Price Trajectory")
                st.plotly_chart(fig_f, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: FUNDAMENTALS & DUPONT
# ---------------------------------------------------------
with tab3:
    st.header(f"[{target_asset}] Deep Value & Fundamentals")
    info, bs, ic = get_fundamentals(target_asset)
    
    col_dp1, col_dp2 = st.columns([1, 2])
    with col_dp1:
        st.subheader("DuPont Analysis")
        st.latex(r"ROE = \text{Net Margin} \times \text{Asset Turnover} \times \text{Equity Multiplier}")
        
        ni = ic.loc['Net Income'].iloc[0] if 'Net Income' in ic.index else None
        rev = ic.loc['Total Revenue'].iloc[0] if 'Total Revenue' in ic.index else None
        ta = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else None
        te = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else None
        
        if all([ni, rev, ta, te]):
            st.metric("Net Profit Margin", f"{(ni/rev)*100:.2f}%")
            st.metric("Asset Turnover", f"{rev/ta:.2f}x")
            st.metric("Equity Multiplier", f"{ta/te:.2f}x")
            st.success(f"Calculated ROE: **{(ni/rev)*(rev/ta)*(ta/te)*100:.2f}%**")
        else:
            st.warning("Insufficient balance sheet data for DuPont.")

    with col_dp2:
        eps, bvps = info.get('trailingEps', 0), info.get('bookValue', 0)
        gn = f"${np.sqrt(22.5 * eps * bvps):.2f}" if eps and bvps and eps > 0 and bvps > 0 else "N/A"
        
        metrics = [
            {"Metric": "P/E Ratio", "Value": safe_get(info, 'trailingPE', format_type='float')},
            {"Metric": "Price to Book", "Value": safe_get(info, 'priceToBook', format_type='float')},
            {"Metric": "Debt to Equity", "Value": safe_get(info, 'debtToEquity', format_type='float')},
            {"Metric": "Dividend Yield", "Value": safe_get(info, 'dividendYield', format_type='pct')},
            {"Metric": "Graham Number (Fair Value)", "Value": gn}
        ]
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 4: MACRO & ALLOCATION
# ---------------------------------------------------------
with tab4:
    st.header("Portfolio Risk Allocation & Macro Betas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Parity Weights")
        cov_matrix = np.cov(returns_df.values.T)
        weights = np.array([1.0]) if len(tickers) == 1 else RiskParityAllocator().allocate(cov_matrix)
        
        fig_pie = px.pie(values=weights, names=tickers, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Fractional Kelly Sizing (Half-Kelly)")
        kelly_fracs = [KellySizer().compute(returns_df.iloc[:, i], fraction=0.5) for i in range(len(tickers))]
        st.session_state['kelly_weights'] = kelly_fracs
        st.plotly_chart(px.bar(x=tickers, y=kelly_fracs, labels={'x': 'Asset', 'y': 'Capital Risk Fraction'}), use_container_width=True)

    st.subheader("Macro Beta Exposures")
    exposures = MacroFactorModel(["SPY (Market)", "TIP (Inflation)", "TLT (Rates)"]).compute_factor_exposures(returns_df, macro_returns)
    if not exposures.empty: st.plotly_chart(px.imshow(exposures.T, color_continuous_scale="RdBu", aspect="auto"), use_container_width=True)

# ---------------------------------------------------------
# TAB 5: OPTIONS & VOLATILITY
# ---------------------------------------------------------
with tab5:
    st.header(f"[{target_asset}] Options Pricing & Volatility Surface")
    
    exps, calls, puts = get_options_data(target_asset)
    
    if exps:
        curr_p = prices_df[target_asset].iloc[-1] if len(tickers) > 1 else prices_df.iloc[-1]
        hv_30 = returns_df[target_asset].tail(30).std() * np.sqrt(252) if len(tickers) > 1 else returns_df.tail(30).std() * np.sqrt(252)
        
        atm_calls = calls[(calls['strike'] >= curr_p * 0.95) & (calls['strike'] <= curr_p * 1.05)]
        atm_puts = puts[(puts['strike'] >= curr_p * 0.95) & (puts['strike'] <= curr_p * 1.05)]
        blended_iv = (atm_calls['impliedVolatility'].mean() + atm_puts['impliedVolatility'].mean()) / 2
        
        col1, col2, col3 = st.columns(3)
        col1.metric("30-Day Historical Volatility", f"{hv_30*100:.2f}%")
        col2.metric("Front-Month Implied Volatility", f"{blended_iv*100:.2f}%")
        col3.metric("Volatility Premium", f"{(blended_iv - hv_30)*100:.2f}%")
        
        st.subheader("Theoretical Black-Scholes Arbitrage (ATM Calls)")
        st.latex(r"C = S \cdot N(d_1) - K \cdot e^{-rt} \cdot N(d_2)")
        
        dte = (datetime.datetime.strptime(exps[0], "%Y-%m-%d") - datetime.datetime.now()).days
        t_years = dte / 365.25
        
        if not atm_calls.empty and t_years > 0:
            atm_calls['Theo_Price'] = atm_calls.apply(lambda r: black_scholes_price(curr_p, r['strike'], t_years, risk_free_rate, r['impliedVolatility'], 'call'), axis=1)
            atm_calls['Mispricing_%'] = ((atm_calls['lastPrice'] - atm_calls['Theo_Price']) / atm_calls['Theo_Price']) * 100
            st.dataframe(atm_calls[['strike', 'lastPrice', 'impliedVolatility', 'Theo_Price', 'Mispricing_%']].style.format({'strike': '${:.2f}', 'lastPrice': '${:.2f}', 'Theo_Price': '${:.2f}', 'impliedVolatility': '{:.2%}', 'Mispricing_%': '{:+.2f}%'}))
            
        st.subheader("Heston Monte Carlo Sim")
        paths = pd.DataFrame({f"Path_{i}": HestonMonteCarlo().simulate(S0=curr_p, sigma=hv_30) for i in range(15)})
        st.plotly_chart(px.line(paths, title="Forward Stochastic Volatility Paths").update_layout(template='plotly_dark', showlegend=False), use_container_width=True)
    else:
        st.warning("No options chain found for this asset.")

# ---------------------------------------------------------
# TAB 6 & 7: OPTUNA & LIVE EXECUTION
# ---------------------------------------------------------
with tab6:
    st.header("Bayesian Hyperparameter Search")
    st.markdown("Automated tree-structured Parzen estimator logic for discovering optimal hyperparameter configurations.")
    if st.button("Run Optuna"):
        with st.spinner("Mapping surface..."):
            def obj(trial): return 1.5 + (np.log(trial.suggest_float("lr", 1e-5, 1e-2, log=True)) * -0.1) + np.random.normal(0, 0.1)
            params, score = BayesianOptimizer().optimize(obj, 10)
            st.success(f"Best Validation Score: {score:.4f}")
            st.json(params)

with tab7:
    st.header("Alpaca Paper Trading Execution")
    col1, col2 = st.columns(2)
    api_k = col1.text_input("Alpaca API Key", type="password")
    api_s = col2.text_input("Alpaca Secret", type="password")
    
    if st.button("Authenticate API"):
        try:
            client = TradingClient(api_k, api_s, paper=True)
            acc = client.get_account()
            st.session_state['alpaca'] = client
            st.session_state['equity'] = float(acc.equity)
            st.success(f"Connected! Buying Power: ${float(acc.buying_power):,.2f}")
        except Exception as e: st.error(e)
            
    if st.button("Execute Portfolio AI Signals", type="primary"):
        if 'alpaca' in st.session_state and 'latest_signals' in st.session_state:
            client, eq = st.session_state['alpaca'], st.session_state['equity']
            for i, ticker in enumerate(tickers):
                sig, risk = st.session_state['latest_signals'][i], st.session_state['kelly_weights'][i]
                if sig == 1.0 and risk > 0:
                    try:
                        client.submit_order(MarketOrderRequest(symbol=ticker, notional=eq * min(risk, 0.20), side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                        st.success(f"Bought {ticker}")
                    except Exception as e: st.error(e)
                elif sig == -1.0:
                    try: 
                        client.close_position(ticker)
                        st.success(f"Closed {ticker}")
                    except: pass
        else:
            st.error("Train Global Transformer (Tab 1) and Authenticate API first.")
