import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score
from plotly import graph_objs as go
import plotly.express as px
import scipy.stats as stats

# Setup hardware acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

st.set_page_config(page_title="Pro Quant Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 1. MODEL ARCHITECTURE (CNN + LSTM + Attention) ---
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector, attn_weights

class HybridQuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.attention = TemporalAttention(128)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.cnn(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)

# --- 2. DATA PIPELINE & HELPERS ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def load_and_process(symbol):
    df = yf.download(symbol, start="2015-01-01", interval="1d", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Support_60d'] = df['Low'].shift(1).rolling(window=60).min()
    df['Resistance_60d'] = df['High'].shift(1).rolling(window=60).max()
    
    return df.dropna().reset_index(drop=True)

@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    tkr = yf.Ticker(symbol)
    info = tkr.info
    try:
        bs = tkr.balance_sheet
        ic = tkr.income_stmt
    except Exception:
        bs = pd.DataFrame()
        ic = pd.DataFrame()
    return info, bs, ic

@st.cache_data(ttl=3600)
def get_options_data(symbol):
    tkr = yf.Ticker(symbol)
    expirations = tkr.options
    if not expirations:
        return None, None, None
    front_month = expirations[0]
    chain = tkr.option_chain(front_month)
    return expirations, chain.calls, chain.puts

def create_sequences(data, features, target_col, s_x, s_y, window):
    x_sc = s_x.transform(data[features])
    y_sc = s_y.transform(data[[target_col]])
    xs, ys = [], []
    for i in range(len(x_sc) - window):
        xs.append(x_sc[i:i+window])
        ys.append(y_sc[i+window])
    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

def calculate_max_drawdown(returns):
    if not returns: return 0.0
    cumulative = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / (running_max + 1e-9)
    return np.min(drawdowns) * 100

def safe_get(d, key, default="N/A", format_type="num"):
    val = d.get(key)
    if val is None or val == "": return default
    if format_type == "pct": return f"{val*100:.2f}%"
    if format_type == "curr": return f"${val:,.2f}"
    if format_type == "float": return f"{val:.2f}"
    return val

def get_financial_metric(df, keys):
    if df.empty: return None
    for k in keys:
        if k in df.index:
            val = df.loc[k].iloc[0]
            if pd.notna(val): return val
    return None

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

# --- 3. UI & DASHBOARD SETUP ---
st.sidebar.header("🕹️ Quantitative Engine")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (for Options):", value=0.045, step=0.005, format="%.3f")
lookback = st.sidebar.slider("Lookback Window:", 10, 60, 30)
epochs = st.sidebar.slider("Training Epochs:", 10, 100, 30)
batch_size = st.sidebar.selectbox("Batch Size:", [32, 64, 128], index=1)
n_splits = st.sidebar.slider("TSCV Folds:", 2, 5, 3)
forecast_horizon = st.sidebar.selectbox("Forecast Horizon:", ["1 Week", "1 Month", "1 Year"])
st.sidebar.divider()
fib_window = st.sidebar.slider("Fibonacci Window (Days):", 30, 1000, 252)

df = load_and_process(ticker)
info, bs, ic = get_fundamentals(ticker)

if df is not None:
    st.title(f"🚀 AI Quant Dashboard: {ticker}")
    st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')} | **Market Cap:** {safe_get(info, 'marketCap', format_type='curr')}")
    st.markdown(f"*Compute Device: {device.type.upper()}*")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Technicals & Forecast", 
        "💎 Deep Value", 
        "🔍 DuPont Analysis", 
        "⚖️ Options & Volatility"
    ])

    # ==========================================
    # TAB 1: TECHNICALS & FORECAST
    # ==========================================
    with tab1:
        st.subheader("Historical Price Action")
        actual_window = min(fib_window, len(df))
        fib_df = df.tail(actual_window)
        max_price = fib_df['High'].max()
        min_price = fib_df['Low'].min()
        start_date = fib_df['Date'].iloc[0]
        end_date = fib_df['Date'].iloc[-1]
        
        diff = max_price - min_price
        fib_levels = {
            "0.0%": min_price, "23.6%": max_price - 0.236 * diff,
            "38.2%": max_price - 0.382 * diff, "50.0%": max_price - 0.5 * diff,
            "61.8%": max_price - 0.618 * diff, "78.6%": max_price - 0.786 * diff,
            "100.0%": max_price
        }

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color="white")))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name="20-Day SMA", line=dict(color="cyan", width=1)))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="50-Day SMA", line=dict(color="orange", width=1)))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Support_60d'], name="60d Support", line=dict(color="red", dash="dash")))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Resistance_60d'], name="60d Resistance", line=dict(color="green", dash="dash")))
        
        colors = ["#ff9999", "#ffcc99", "#ffff99", "#ccff99", "#99ff99", "#99ccff", "#cc99ff"]
        for (level_name, price), color in zip(fib_levels.items(), colors):
            fig_hist.add_trace(go.Scatter(
                x=[start_date, end_date], y=[price, price], mode="lines+text",
                name=f"Fib {level_name}", line=dict(color=color, dash="dot"),
                text=[None, f"{level_name}"], textposition="top left", showlegend=False
            ))

        fig_hist.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

        if st.button("🔄 Run AI Model Pipeline & Generate Forecast"):
            features = ['Log_Ret', 'Vol_20', 'RSI']
            target_col = 'Log_Ret'
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            y_actual_all, y_pred_ai_all, y_pred_ma_all = [], [], []
            strat_rets_ai_all, strat_rets_ma_all = [], []
            last_train_df, last_test_df = None, None

            with st.status("Training CNN-LSTM Attention Model...", expanded=True) as status:
                for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
                    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
                    if i == n_splits - 1: last_train_df, last_test_df = train_df, test_df

                    sc_x = RobustScaler().fit(train_df[features])
                    sc_y = RobustScaler().fit(train_df[[target_col]])
                    
                    X_train, y_train = create_sequences(train_df, features, target_col, sc_x, sc_y, lookback)
                    X_test, y_test = create_sequences(test_df, features, target_col, sc_x, sc_y, lookback)
                    
                    train_dataset = TensorDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    
                    model = HybridQuantModel(len(features)).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
                    criterion = nn.HuberLoss()
                    
                    for epoch in range(epochs):
                        model.train()
                        epoch_loss = 0
                        for batch_x, batch_y in train_loader:
                            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                            optimizer.zero_grad()
                            loss = criterion(model(batch_x), batch_y) 
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                        scheduler.step(epoch_loss)
                    
                    model.eval()
                    with torch.no_grad():
                        X_test_dev = X_test.to(device)
                        y_pred_scaled = model(X_test_dev).cpu().numpy()
                        y_pred_ai = sc_y.inverse_transform(y_pred_scaled).flatten()
                        y_actual = sc_y.inverse_transform(y_test.numpy()).flatten()
                        
                        ma_signals = np.where(test_df['SMA_20'].iloc[lookback:].values > test_df['SMA_50'].iloc[lookback:].values, 1, -1)
                        min_len = min(len(y_actual), len(ma_signals))
                        y_actual, y_pred_ai, ma_signals = y_actual[:min_len], y_pred_ai[:min_len], ma_signals[:min_len]

                        y_actual_all.extend(y_actual)
                        y_pred_ai_all.extend(y_pred_ai)
                        y_pred_ma_all.extend(ma_signals)
                        
                        ai_rets = np.where(y_pred_ai > 0, 1, -1) * y_actual
                        ma_rets = ma_signals * y_actual
                        
                        strat_rets_ai_all.extend(ai_rets)
                        strat_rets_ma_all.extend(ma_rets)
                        
                        sharpe_ai = (np.mean(ai_rets) / (np.std(ai_rets) + 1e-9)) * np.sqrt(252)
                        st.write(f"✅ Fold {i+1} Validated | OOS Sharpe Ratio: **{sharpe_ai:.2f}**")
                
                st.write("🌐 Training Final Production Model on Full Dataset...")
                prod_model = HybridQuantModel(len(features)).to(device)
                opt_f = torch.optim.AdamW(prod_model.parameters(), lr=0.001)
                
                sc_x_f = RobustScaler().fit(df[features])
                sc_y_f = RobustScaler().fit(df[[target_col]])
                X_f, y_f = create_sequences(df, features, target_col, sc_x_f, sc_y_f, lookback)
                
                prod_dataset = TensorDataset(X_f, y_f)
                prod_loader = DataLoader(prod_dataset, batch_size=batch_size, shuffle=True)
                
                for _ in range(epochs):
                    prod_model.train()
                    for batch_x, batch_y in prod_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        opt_f.zero_grad()
                        nn.HuberLoss()(prod_model(batch_x), batch_y).backward()
                        opt_f.step()

                prod_model.eval()
                current_win_unscaled = df[features].tail(lookback).values
                forecast_prices = [df['Close'].iloc[-1]]
                forecast_dates = [df['Date'].iloc[-1]]
                h_days = {'1 Week': 5, '1 Month': 21, '1 Year': 252}[forecast_horizon]
                
                recent_prices = list(df['Close'].tail(max(20, lookback)).values)
                recent_rets = list(df['Log_Ret'].tail(max(20, lookback)).values)
                
                for _ in range(h_days):
                    win_scaled = sc_x_f.transform(current_win_unscaled)
                    win_t = torch.FloatTensor(win_scaled).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred_scaled = prod_model(win_t).cpu().item()
                        p_ret = sc_y_f.inverse_transform([[pred_scaled]])[0][0]
                    
                    if forecast_horizon == '1 Year':
                        p_ret = p_ret * 0.95 
                    
                    next_price = forecast_prices[-1] * np.exp(p_ret)
                    forecast_prices.append(next_price)
                    recent_prices.append(next_price)
                    recent_rets.append(p_ret)
                    
                    next_date = forecast_dates[-1] + timedelta(days=1)
                    while next_date.weekday() >= 5: next_date += timedelta(days=1)
                    forecast_dates.append(next_date)
                    
                    new_vol = np.std(recent_rets[-20:])
                    new_rsi = calculate_rsi(pd.Series(recent_prices[-15:]), 14).iloc[-1]
                    new_row = np.array([p_ret, new_vol, new_rsi])
                    current_win_unscaled = np.append(current_win_unscaled[1:], [new_row], axis=0)

                status.update(label="Validation & Forecasting Complete!", state="complete")

            st.subheader(f"🔮 AI Price Forecast ({forecast_horizon})")
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=last_train_df['Date'], y=last_train_df['Close'], name="Train Data", line=dict(color="#1f77b4")))
            fig_f.add_trace(go.Scatter(x=last_test_df['Date'], y=last_test_df['Close'], name="Test Data (OOS)", line=dict(color="#ff7f0e")))
            fig_f.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, name="LSTM Forecast", line=dict(color="#2ca02c", width=3, dash="dash")))
            fig_f.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_f, use_container_width=True)

            col1, col2 = st.columns(2)
            y_act_dir = (np.array(y_actual_all) > 0).astype(int)
            y_pred_ai_dir = (np.array(y_pred_ai_all) > 0).astype(int)
            y_pred_ma_dir = (np.array(y_pred_ma_all) > 0).astype(int)

            with col1:
                st.markdown("### 🤖 CNN-LSTM Model Stats")
                st.metric("Directional F1", f"{f1_score(y_act_dir, y_pred_ai_dir):.4f}")
                st.metric("Cumulative Return", f"{np.sum(strat_rets_ai_all)*100:.2f}%")
            with col2:
                st.markdown("### 📊 Baseline (20/50 SMA)")
                st.metric("Directional F1", f"{f1_score(y_act_dir, y_pred_ma_dir):.4f}")
                st.metric("Cumulative Return", f"{np.sum(strat_rets_ma_all)*100:.2f}%")

    # ==========================================
    # TAB 2: DEEP VALUE METRICS
    # ==========================================
    with tab2:
        st.subheader("💎 Key Fundamental Metrics")
        metrics = [
            ("P/E Ratio", safe_get(info, 'trailingPE', format_type='float')),
            ("Forward P/E", safe_get(info, 'forwardPE', format_type='float')),
            ("Price to Book (P/B)", safe_get(info, 'priceToBook', format_type='float')),
            ("EV to EBITDA", safe_get(info, 'enterpriseToEbitda', format_type='float')),
            ("Debt to Equity", safe_get(info, 'debtToEquity', format_type='float')),
            ("Return on Equity (ROE)", safe_get(info, 'returnOnEquity', format_type='pct')),
        ]
        col1, col2, col3 = st.columns(3)
        for i, (name, val) in enumerate(metrics):
            if i % 3 == 0: col1.metric(name, val)
            elif i % 3 == 1: col2.metric(name, val)
            else: col3.metric(name, val)

    # ==========================================
    # TAB 3: DUPONT ANALYSIS
    # ==========================================
    with tab3:
        st.subheader("🔍 DuPont Analysis")
        st.latex(r"ROE = \text{Net Profit Margin} \times \text{Asset Turnover} \times \text{Equity Multiplier}")
        
        net_income = get_financial_metric(ic, ['Net Income', 'Net Income Common Stockholders'])
        revenue = get_financial_metric(ic, ['Total Revenue', 'Operating Revenue'])
        total_assets = get_financial_metric(bs, ['Total Assets'])
        total_equity = get_financial_metric(bs, ['Stockholders Equity', 'Total Stockholder Equity'])
        
        if all([net_income, revenue, total_assets, total_equity]):
            npm = net_income / revenue
            ato = revenue / total_assets
            em = total_assets / total_equity
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Profit Margin", f"{npm*100:.2f}%")
            c2.metric("Asset Turnover", f"{ato:.2f}x")
            c3.metric("Equity Multiplier", f"{em:.2f}x")
            c4.metric("Calculated ROE", f"{npm * ato * em * 100:.2f}%")
        else:
            st.warning("Insufficient fundamental data available for this ticker.")

    # ==========================================
    # TAB 4: OPTIONS & VOLATILITY
    # ==========================================
    with tab4:
        st.subheader("⚖️ Volatility Surface & Actionable Strategies")
        exps, calls, puts = get_options_data(ticker)
        
        if exps is None:
            st.warning("No options data available for this ticker.")
        else:
            current_price = df['Close'].iloc[-1]
            hv_30 = df['Log_Ret'].tail(30).std() * np.sqrt(252)
            
            atm_calls = calls[(calls['strike'] >= current_price * 0.95) & (calls['strike'] <= current_price * 1.05)]
            atm_puts = puts[(puts['strike'] >= current_price * 0.95) & (puts['strike'] <= current_price * 1.05)]
            blended_iv = (atm_calls['impliedVolatility'].mean() + atm_puts['impliedVolatility'].mean()) / 2
            vol_premium = blended_iv - hv_30
            
            c1, c2, c3 = st.columns(3)
            c1.metric("30-Day Historical Volatility (HV)", f"{hv_30*100:.2f}%")
            c2.metric("Front-Month Implied Volatility (IV)", f"{blended_iv*100:.2f}%")
            c3.metric("Volatility Premium (IV - HV)", f"{vol_premium*100:.2f}%")

            st.divider()
            
            st.subheader("📐 Black-Scholes Theoretical Pricing (ATM Calls)")
            st.latex(r"C = S \cdot N(d_1) - K \cdot e^{-rt} \cdot N(d_2)")
            
            # Calculate Time to Expiration in Years
            exp_date = datetime.strptime(exps[0], "%Y-%m-%d")
            dte = (exp_date - datetime.now()).days
            time_to_exp_years = dte / 365.25
            
            if not atm_calls.empty and time_to_exp_years > 0:
                atm_calls['Theoretical_Price'] = atm_calls.apply(
                    lambda row: black_scholes_price(current_price, row['strike'], time_to_exp_years, risk_free_rate, row['impliedVolatility'], 'call'), axis=1
                )
                atm_calls['Mispricing_%'] = ((atm_calls['lastPrice'] - atm_calls['Theoretical_Price']) / atm_calls['Theoretical_Price']) * 100
                
                disp_cols = ['contractSymbol', 'strike', 'lastPrice', 'impliedVolatility', 'Theoretical_Price', 'Mispricing_%']
                st.dataframe(atm_calls[disp_cols].style.format({
                    'strike': '${:.2f}', 'lastPrice': '${:.2f}', 'Theoretical_Price': '${:.2f}',
                    'impliedVolatility': '{:.2%}', 'Mispricing_%': '{:+.2f}%'
                }), use_container_width=True)
            else:
                st.info("Options expire today or not enough ATM liquidity for theoretical pricing.")

            st.divider()

            current_sma20, current_sma50, current_rsi = df['SMA_20'].iloc[-1], df['SMA_50'].iloc[-1], df['RSI'].iloc[-1]
            trend = "Bullish" if current_sma20 > current_sma50 else "Bearish"
            momentum = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"

            st.subheader(f"🤖 Algorithmic Strategy ({trend} / {momentum})")
            
            if vol_premium > 0.05:
                strat = "Bull Put Spread" if trend == "Bullish" and current_rsi < 70 else "Bear Call Spread" if trend == "Bearish" and current_rsi > 30 else "Iron Condor"
            elif vol_premium < -0.02:
                strat = "Long Call Spread" if trend == "Bullish" and current_rsi < 70 else "Long Put Spread" if trend == "Bearish" and current_rsi > 30 else "Long Straddle"
            else:
                strat = "Covered Call" if trend == "Bullish" else "Cash Secured Put"

            st.success(f"**Recommended Action:** {strat}")
            
            # NEW: Strike Recommendation Logic added here
            if time_to_exp_years > 0:
                expected_move = current_price * blended_iv * np.sqrt(time_to_exp_years)
                upper_1sd = current_price + expected_move
                lower_1sd = current_price - expected_move
                upper_05sd = current_price + (expected_move * 0.5)
                lower_05sd = current_price - (expected_move * 0.5)
                
                if strat in ["Bull Put Spread", "Cash Secured Put"]:
                    st.info(f"🎯 **Strike Selection (Puts):** For higher win rate (~68%), target short strikes near **${lower_1sd:.2f}**. For maximum yield/premium, target closer to **${lower_05sd:.2f}**.")
                elif strat in ["Bear Call Spread", "Covered Call"]:
                    st.info(f"🎯 **Strike Selection (Calls):** For higher win rate (~68%), target short strikes near **${upper_1sd:.2f}**. For maximum yield/premium, target closer to **${upper_05sd:.2f}**.")
                elif strat == "Iron Condor":
                    st.info(f"🎯 **Strike Selection (Condor):** For higher win rate, sell wings outside **${lower_1sd:.2f}** (Put) and **${upper_1sd:.2f}** (Call). For max yield, bring short strikes closer to **${lower_05sd:.2f}** and **${upper_05sd:.2f}**.")
                elif strat in ["Long Call Spread", "Long Put Spread", "Long Straddle"]:
                    st.info(f"🎯 **Strike Selection (Directional/Debit):** For higher win rate, buy At-The-Money (ATM) near **${current_price:.2f}**. For max theoretical yield (ROI), buy Out-Of-The-Money (OTM) near **${upper_05sd:.2f}** (Calls) or **${lower_05sd:.2f}** (Puts).")
else:
    st.error("Ticker not found.")
