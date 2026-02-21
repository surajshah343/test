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

# Setup hardware acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

st.set_page_config(page_title="Pro Quant Dashboard", layout="wide")

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
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector, attn_weights

class HybridQuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Spatial Feature Extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64)
        )
        # Temporal Processing
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        # Attention Mechanism
        self.attention = TemporalAttention(128)
        # Fully Connected Regression Head
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # CNN expects (batch, channels, length)
        x = x.transpose(1, 2) 
        x = self.cnn(x)
        # LSTM expects (batch, length, features)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)

# --- 2. DATA PIPELINE & STRICT NO-LEAKAGE HELPERS ---
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
    
    # Technical Features (Calculated on current day 't')
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # Baseline Strategy Features
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Support & Resistance (Strict Leakage Prevention: Shift by 1)
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

def create_sequences(data, features, target_col, s_x, s_y, window):
    x_sc = s_x.transform(data[features])
    y_sc = s_y.transform(data[[target_col]])
    xs, ys = [], []
    for i in range(len(x_sc) - window):
        xs.append(x_sc[i:i+window])
        # Target natively aligns with the period *following* the window
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

# --- 3. UI & DASHBOARD SETUP ---
st.sidebar.header("🕹️ Quantitative Engine")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
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
    
    tab1, tab2, tab3 = st.tabs(["📈 Technicals & Forecast", "💎 Deep Value", "🔍 DuPont Analysis"])

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
            "0.0% (Low Wick)": min_price,
            "23.6%": max_price - 0.236 * diff,
            "38.2%": max_price - 0.382 * diff,
            "50.0%": max_price - 0.5 * diff,
            "61.8%": max_price - 0.618 * diff,
            "78.6%": max_price - 0.786 * diff,
            "100.0% (High Wick)": max_price
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
                x=[start_date, end_date], 
                y=[price, price], 
                mode="lines+text",
                name=f"Fib {level_name}",
                line=dict(color=color, dash="dot"),
                text=[None, f"{level_name}"],
                textposition="top left",
                showlegend=False
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

            with st.status("Training CNN-LSTM Attention Model (Strict No-Leakage Policy)...", expanded=True) as status:
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
                        
                        # Align MA signals
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
                
                # --- PRODUCTION MODEL & FORECAST ---
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
                    # Scale current window for inference
                    win_scaled = sc_x_f.transform(current_win_unscaled)
                    win_t = torch.FloatTensor(win_scaled).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred_scaled = prod_model(win_t).cpu().item()
                        p_ret = sc_y_f.inverse_transform([[pred_scaled]])[0][0]
                    
                    if forecast_horizon == '1 Year':
                        p_ret = p_ret * 0.95 # Autoregressive dampening
                    
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
                    # Shift unscaled window
                    current_win_unscaled = np.append(current_win_unscaled[1:], [new_row], axis=0)

                status.update(label="Validation & Forecasting Complete!", state="complete")

            # --- TRAIN / TEST / FORECAST PLOT ---
            st.subheader(f"🔮 AI Price Forecast ({forecast_horizon}) & Final Fold OOS Check")
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=last_train_df['Date'], y=last_train_df['Close'], name="Train Data", line=dict(color="#1f77b4")))
            fig_f.add_trace(go.Scatter(x=last_test_df['Date'], y=last_test_df['Close'], name="Test Data (OOS)", line=dict(color="#ff7f0e")))
            fig_f.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, name="LSTM Forecast", line=dict(color="#2ca02c", width=3, dash="dash")))
            fig_f.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_f, use_container_width=True)

            # --- FORECAST ACCURACY ---
            st.subheader("🎯 Out-of-Sample Accuracy: Advanced DL vs Moving Average Baseline")
            y_act_dir = (np.array(y_actual_all) > 0).astype(int)
            y_pred_ai_dir = (np.array(y_pred_ai_all) > 0).astype(int)
            y_pred_ma_dir = (np.array(y_pred_ma_all) > 0).astype(int)

            f1_ai = f1_score(y_act_dir, y_pred_ai_dir)
            f1_ma = f1_score(y_act_dir, y_pred_ma_dir)
            cum_ret_ai = np.sum(strat_rets_ai_all) * 100
            cum_ret_ma = np.sum(strat_rets_ma_all) * 100
            dd_ai = calculate_max_drawdown(strat_rets_ai_all)
            dd_ma = calculate_max_drawdown(strat_rets_ma_all)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🤖 CNN-LSTM Attention Model")
                st.metric("Directional F1-Score", f"{f1_ai:.4f}")
                st.metric("Cumulative OOS Return", f"{cum_ret_ai:.2f}%")
                st.metric("Maximum Drawdown", f"{dd_ai:.2f}%")
                st.metric("RMSE (Log Ret)", f"{np.sqrt(mean_squared_error(y_actual_all, y_pred_ai_all)):.4f}")
                
            with col2:
                st.markdown("### 📊 Baseline (20/50 SMA)")
                st.metric("Directional F1-Score", f"{f1_ma:.4f}", delta=f"{f1_ma - f1_ai:.4f} vs AI", delta_color="normal")
                st.metric("Cumulative OOS Return", f"{cum_ret_ma:.2f}%", delta=f"{cum_ret_ma - cum_ret_ai:.2f}% vs AI", delta_color="normal")
                st.metric("Maximum Drawdown", f"{dd_ma:.2f}%", delta=f"{dd_ma - dd_ai:.2f}% vs AI", delta_color="inverse")
                st.metric("RMSE (Log Ret)", "N/A")

            # --- FEATURE IMPORTANCE ---
            st.subheader("🧠 Permutation Feature Importance (Spatial)")
            importances = []
            with torch.no_grad():
                X_test_dev = X_test.to(device)
                baseline_loss = nn.HuberLoss()(model(X_test_dev), y_test.to(device)).item()
                for f_idx in range(len(features)):
                    X_temp = X_test_dev.clone()
                    X_temp[:, :, f_idx] = X_temp[torch.randperm(X_temp.size(0)), :, f_idx]
                    shuffled_loss = nn.HuberLoss()(model(X_temp), y_test.to(device)).item()
                    importances.append(max(0, shuffled_loss - baseline_loss))
            
            imp_df = pd.DataFrame({'Feature': features, 'Impact': importances}).sort_values(by='Impact', ascending=True)
            fig_imp = px.bar(imp_df, x='Impact', y='Feature', orientation='h', template="plotly_dark", title="Loss Increase When Feature is Shuffled")
            st.plotly_chart(fig_imp, use_container_width=True)

    # ==========================================
    # TAB 2: DEEP VALUE METRICS
    # ==========================================
    with tab2:
        st.subheader("💎 20 Deep Value & Solvency Metrics")
        eps_val = info.get('trailingEps')
        bvps_val = info.get('bookValue')
        if eps_val and bvps_val and eps_val > 0 and bvps_val > 0:
            graham_num = np.sqrt(22.5 * eps_val * bvps_val)
            graham_val_str = f"${graham_num:.2f}"
        else:
            graham_val_str = "N/A"

        metrics_data = [
            ("P/E Ratio", safe_get(info, 'trailingPE', format_type='float'), r"\frac{\text{Market Price}}{\text{Trailing EPS}}", "TTM"),
            ("Forward P/E", safe_get(info, 'forwardPE', format_type='float'), r"\frac{\text{Market Price}}{\text{Estimated Future EPS}}", "Next 12M"),
            ("PEG Ratio", safe_get(info, 'pegRatio', format_type='float'), r"\frac{\text{P/E Ratio}}{\text{Earnings Growth Rate}}", "5Y Expected"),
            ("Graham Number", graham_val_str, r"\sqrt{22.5 \times \text{EPS} \times \text{BVPS}}", "TTM/MRQ"),
            ("Price to Book (P/B)", safe_get(info, 'priceToBook', format_type='float'), r"\frac{\text{Market Price}}{\text{Book Value per Share}}", "MRQ"),
            ("Price to Sales (P/S)", safe_get(info, 'priceToSalesTrailing12Months', format_type='float'), r"\frac{\text{Market Cap}}{\text{Total Revenue}}", "TTM"),
            ("EV to EBITDA", safe_get(info, 'enterpriseToEbitda', format_type='float'), r"\frac{\text{Enterprise Value}}{\text{EBITDA}}", "TTM"),
            ("EV to Revenue", safe_get(info, 'enterpriseToRevenue', format_type='float'), r"\frac{\text{Enterprise Value}}{\text{Revenue}}", "TTM"),
            ("Dividend Yield", safe_get(info, 'dividendYield', format_type='pct'), r"\frac{\text{Annual Dividends per Share}}{\text{Price per Share}}", "Forward"),
            ("Payout Ratio", safe_get(info, 'payoutRatio', format_type='pct'), r"\frac{\text{Dividends Paid}}{\text{Net Income}}", "TTM"),
            ("Current Ratio", safe_get(info, 'currentRatio', format_type='float'), r"\frac{\text{Current Assets}}{\text{Current Liabilities}}", "MRQ"),
            ("Quick Ratio", safe_get(info, 'quickRatio', format_type='float'), r"\frac{\text{Current Assets} - \text{Inventory}}{\text{Current Liabilities}}", "MRQ"),
            ("Debt to Equity", safe_get(info, 'debtToEquity', format_type='float'), r"\frac{\text{Total Liabilities}}{\text{Shareholders' Equity}}", "MRQ"),
            ("Return on Equity (ROE)", safe_get(info, 'returnOnEquity', format_type='pct'), r"\frac{\text{Net Income}}{\text{Shareholders' Equity}}", "TTM"),
            ("Return on Assets (ROA)", safe_get(info, 'returnOnAssets', format_type='pct'), r"\frac{\text{Net Income}}{\text{Total Assets}}", "TTM"),
            ("Gross Margin", safe_get(info, 'grossMargins', format_type='pct'), r"\frac{\text{Revenue} - \text{COGS}}{\text{Revenue}}", "TTM"),
            ("Operating Margin", safe_get(info, 'operatingMargins', format_type='pct'), r"\frac{\text{Operating Income}}{\text{Revenue}}", "TTM"),
        ]

        formatted_metrics = []
        for name, val, formula, period in metrics_data:
            formatted_metrics.append((name, val, formula, period))

        for i in range(0, len(formatted_metrics), 2):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{formatted_metrics[i][0]}** | Period: *{formatted_metrics[i][3]}*")
                st.latex(formatted_metrics[i][2])
                st.markdown(f"> **Value: {formatted_metrics[i][1]}**")
                st.divider()
            if i + 1 < len(formatted_metrics):
                with col2:
                    st.markdown(f"**{formatted_metrics[i+1][0]}** | Period: *{formatted_metrics[i+1][3]}*")
                    st.latex(formatted_metrics[i+1][2])
                    st.markdown(f"> **Value: {formatted_metrics[i+1][1]}**")
                    st.divider()

    # ==========================================
    # TAB 3: DUPONT ANALYSIS
    # ==========================================
    with tab3:
        
        st.subheader("🔍 3-Step DuPont Analysis")
        st.latex(r"ROE = \left( \frac{\text{Net Income}}{\text{Sales}} \right) \times \left( \frac{\text{Sales}}{\text{Total Assets}} \right) \times \left( \frac{\text{Total Assets}}{\text{Total Equity}} \right)")
        
        try:
            net_income_keys = ['Net Income', 'Net Income Common Stockholders']
            revenue_keys = ['Total Revenue', 'Operating Revenue']
            assets_keys = ['Total Assets']
            equity_keys = ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']
            
            net_income = get_financial_metric(ic, net_income_keys)
            revenue = get_financial_metric(ic, revenue_keys)
            total_assets = get_financial_metric(bs, assets_keys)
            total_equity = get_financial_metric(bs, equity_keys)
            
            if all([net_income, revenue, total_assets, total_equity]):
                npm = net_income / revenue
                ato = revenue / total_assets
                em = total_assets / total_equity
                calculated_roe = npm * ato * em

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Net Profit Margin", f"{npm*100:.2f}%")
                col2.metric("Asset Turnover", f"{ato:.2f}x")
                col3.metric("Equity Multiplier (Leverage)", f"{em:.2f}x")
                col4.metric("Calculated ROE", f"{calculated_roe*100:.2f}%", delta="DuPont Result")
                
                st.info("💡 **Interpretation:** High ROE driven by Net Profit Margin is highly desirable (pricing power). If driven mostly by the Equity Multiplier, the company relies heavily on debt.")
            else:
                st.warning("Insufficient balance sheet or income statement data available for this ticker.")
        except Exception:
            st.error("Error calculating manual DuPont breakdown.")
else:
    st.error("Ticker not found. Please check the symbol.")
