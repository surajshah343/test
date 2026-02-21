import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotly import graph_objs as go
import plotly.express as px

st.set_page_config(page_title="Pro Quant Dashboard", layout="wide")

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

# --- 2. DATA PIPELINE & HELPERS ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def load_and_process(symbol):
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    # Technical Features (Strictly Historical)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # Baseline Strategy Features (Rolling SMAs)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Support & Resistance (Rolling 60-day min/max to prevent leakage)
    df['Support_60d'] = df['Low'].rolling(window=60).min()
    df['Resistance_60d'] = df['High'].rolling(window=60).max()
    
    # Target: Next day's return
    df['Target'] = df['Log_Ret'].shift(-1)
    return df.dropna().reset_index(drop=True)

@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    tkr = yf.Ticker(symbol)
    info = tkr.info
    try:
        bs = tkr.balance_sheet
        ic = tkr.income_stmt
    except:
        bs = pd.DataFrame()
        ic = pd.DataFrame()
    return info, bs, ic

def create_sequences(data, features, target_col, s_x, s_y, window):
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

# --- 3. UI & DASHBOARD SETUP ---
st.sidebar.header("🕹️ Strategy Engine")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
lookback = st.sidebar.slider("Lookback Window:", 10, 60, 30)
epochs = st.sidebar.slider("Training Epochs:", 5, 50, 20)
n_splits = st.sidebar.slider("TSCV Folds:", 2, 5, 3)
forecast_horizon = st.sidebar.selectbox("Forecast Horizon:", ["1 Week", "1 Month", "1 Year"])

df = load_and_process(ticker)
info, bs, ic = get_fundamentals(ticker)

if df is not None:
    st.title(f"🚀 AI Quant Dashboard: {ticker}")
    st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')} | **Market Cap:** {safe_get(info, 'marketCap', format_type='curr')}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Technicals, AI vs Baseline & Forecast", "💎 Deep Value Metrics", "🔍 DuPont Analysis"])

    # ==========================================
    # TAB 1: TECHNICALS & FORECAST
    # ==========================================
    with tab1:
        st.subheader("Historical Price Action with Support/Resistance & Fibonacci")
        
        # Calculate Fibonacci Retracement Levels
        max_price = df['High'].max()
        min_price = df['Low'].min()
        diff = max_price - min_price
        fib_levels = {
            "0.0% (Low)": min_price,
            "23.6%": max_price - 0.236 * diff,
            "38.2%": max_price - 0.382 * diff,
            "50.0%": max_price - 0.5 * diff,
            "61.8%": max_price - 0.618 * diff,
            "78.6%": max_price - 0.786 * diff,
            "100.0% (High)": max_price
        }

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color="white")))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name="20-Day SMA", line=dict(color="cyan", width=1)))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="50-Day SMA", line=dict(color="orange", width=1)))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Support_60d'], name="60d Support", line=dict(color="red", dash="dash")))
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Resistance_60d'], name="60d Resistance", line=dict(color="green", dash="dash")))
        
        # Add Fibonacci Lines
        colors = ["#ff9999", "#ffcc99", "#ffff99", "#ccff99", "#99ff99", "#99ccff", "#cc99ff"]
        for (level_name, price), color in zip(fib_levels.items(), colors):
            fig_hist.add_hline(y=price, line_dash="dot", line_color=color, annotation_text=level_name, annotation_position="top left")

        fig_hist.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

        if st.button("🔄 Run AI Model Pipeline & Generate Forecast"):
            features = ['Log_Ret', 'Vol_20', 'RSI']
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            y_actual_all, y_pred_ai_all, y_pred_ma_all = [], [], []
            strat_rets_ai_all, strat_rets_ma_all = [], []
            
            last_train_df, last_test_df = None, None

            with st.status("Training AI & Validating Against MA Crossover (Strict No-Leakage Policy)...", expanded=True) as status:
                for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
                    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
                    
                    # Save last fold for visualization
                    if i == n_splits - 1:
                        last_train_df, last_test_df = train_df, test_df

                    # STRICT LEAKAGE PREVENTION: Fit scalers ONLY on training data
                    sc_x = RobustScaler().fit(train_df[features])
                    sc_y = RobustScaler().fit(train_df[['Target']])
                    
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
                        y_pred_scaled = model(X_test).numpy()
                        y_pred_ai = sc_y.inverse_transform(y_pred_scaled).flatten()
                        y_actual = sc_y.inverse_transform(y_test.numpy()).flatten()
                        
                        ma_signals = np.where(test_df['SMA_20'].iloc[lookback-1:-1].values > test_df['SMA_50'].iloc[lookback-1:-1].values, 1, 0)
                        
                        y_actual_all.extend(y_actual)
                        y_pred_ai_all.extend(y_pred_ai)
                        y_pred_ma_all.extend(ma_signals)
                        
                        ai_rets = np.where(y_pred_ai > 0, 1, 0) * y_actual
                        ma_rets = ma_signals * y_actual
                        
                        strat_rets_ai_all.extend(ai_rets)
                        strat_rets_ma_all.extend(ma_rets)
                        
                        sharpe_ai = (np.mean(ai_rets) / (np.std(ai_rets) + 1e-9)) * np.sqrt(252)
                        sharpe_ma = (np.mean(ma_rets) / (np.std(ma_rets) + 1e-9)) * np.sqrt(252)
                        
                        st.write(f"✅ Fold {i+1} Validated | AI Sharpe: **{sharpe_ai:.2f}** vs MA Sharpe: **{sharpe_ma:.2f}**")
                
                # --- PRODUCTION MODEL & FORECAST ---
                st.write("🌐 Training Final Production Model on All Data...")
                prod_model = HybridQuantModel(len(features))
                opt_f = torch.optim.Adam(prod_model.parameters(), lr=0.001)
                sc_x_f = RobustScaler().fit(df[features])
                sc_y_f = RobustScaler().fit(df[['Target']])
                X_f, y_f = create_sequences(df, features, 'Target', sc_x_f, sc_y_f, lookback)
                
                for _ in range(epochs):
                    prod_model.train()
                    opt_f.zero_grad()
                    nn.MSELoss()(prod_model(X_f), y_f).backward()
                    opt_f.step()

                prod_model.eval()
                current_win = df[features].tail(lookback).values
                forecast_prices = [df['Close'].iloc[-1]]
                forecast_dates = [df['Date'].iloc[-1]]
                
                h_days = {'1 Week': 5, '1 Month': 21, '1 Year': 252}[forecast_horizon]
                
                # FIX: Maintain active lists of recent valid prices/returns to accurately compute features
                recent_prices = list(df['Close'].tail(max(20, lookback)).values)
                recent_rets = list(df['Log_Ret'].tail(max(20, lookback)).values)
                
                for _ in range(h_days):
                    win_t = torch.FloatTensor(sc_x_f.transform(current_win)).unsqueeze(0)
                    with torch.no_grad():
                        p_ret = sc_y_f.inverse_transform([[prod_model(win_t).item()]])[0][0]
                    
                    # Apply a slight autoregressive dampening factor for longer horizons to prevent runaway compounding errors
                    if forecast_horizon == '1 Year':
                        p_ret = p_ret * 0.90
                    
                    next_price = forecast_prices[-1] * np.exp(p_ret)
                    forecast_prices.append(next_price)
                    
                    # Update active lists
                    recent_prices.append(next_price)
                    recent_rets.append(p_ret)
                    
                    # Add business day
                    next_date = forecast_dates[-1] + timedelta(days=1)
                    while next_date.weekday() >= 5:
                        next_date += timedelta(days=1)
                    forecast_dates.append(next_date)
                    
                    # FIX: Correctly calculate Volatility and RSI on the dynamically built arrays
                    new_vol = np.std(recent_rets[-20:])
                    new_rsi = calculate_rsi(pd.Series(recent_prices[-15:]), 14).iloc[-1]
                    
                    new_row = np.array([p_ret, new_vol, new_rsi])
                    current_win = np.append(current_win[1:], [new_row], axis=0)

                status.update(label="Validation & Forecasting Complete!", state="complete")

            # --- TRAIN / TEST / FORECAST PLOT ---
            st.subheader(f"🔮 AI Price Forecast ({forecast_horizon}) & Fold Evaluation Split")
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=last_train_df['Date'], y=last_train_df['Close'], name="Train Data (Last Fold)", line=dict(color="#1f77b4")))
            fig_f.add_trace(go.Scatter(x=last_test_df['Date'], y=last_test_df['Close'], name="Test Data (OOS)", line=dict(color="#ff7f0e")))
            fig_f.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, name="AI Forecast", line=dict(color="#2ca02c", width=3, dash="dash")))
            
            fig_f.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_f, use_container_width=True)

            # --- FORECAST ACCURACY ---
            st.subheader("🎯 Out-of-Sample Accuracy: AI Model vs Moving Average")
            hit_rate_ai = np.mean((np.array(y_pred_ai_all) > 0) == (np.array(y_actual_all) > 0)) * 100
            hit_rate_ma = np.mean(np.array(y_pred_ma_all) == (np.array(y_actual_all) > 0).astype(int)) * 100
            cum_ret_ai = np.sum(strat_rets_ai_all) * 100
            cum_ret_ma = np.sum(strat_rets_ma_all) * 100

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🤖 Hybrid AI Model")
                st.metric("Directional Hit Rate", f"{hit_rate_ai:.2f}%")
                st.metric("Cumulative Out-of-Sample Return", f"{cum_ret_ai:.2f}%")
                st.metric("RMSE (Log Ret)", f"{np.sqrt(mean_squared_error(y_actual_all, y_pred_ai_all)):.4f}")
                
            with col2:
                st.markdown("### 📊 Baseline (20/50 SMA)")
                st.metric("Directional Hit Rate", f"{hit_rate_ma:.2f}%", delta=f"{hit_rate_ma - hit_rate_ai:.2f}% vs AI", delta_color="normal")
                st.metric("Cumulative Out-of-Sample Return", f"{cum_ret_ma:.2f}%", delta=f"{cum_ret_ma - cum_ret_ai:.2f}% vs AI", delta_color="normal")
                st.metric("RMSE (Log Ret)", "N/A (Signal Only)")

            # --- FEATURE IMPORTANCE ---
            st.subheader("🧠 AI Permutation Feature Importance")
            importances = []
            with torch.no_grad():
                baseline_loss = nn.MSELoss()(model(X_test), y_test).item()
                for f_idx in range(len(features)):
                    X_temp = X_test.clone()
                    X_temp[:, :, f_idx] = X_temp[torch.randperm(X_temp.size(0)), :, f_idx]
                    shuffled_loss = nn.MSELoss()(model(X_temp), y_test).item()
                    importances.append(max(0, shuffled_loss - baseline_loss))
            
            imp_df = pd.DataFrame({'Feature': features, 'Impact': importances}).sort_values(by='Impact', ascending=True)
            fig_imp = px.bar(imp_df, x='Impact', y='Feature', orientation='h', template="plotly_dark", title="MSE Increase When Feature is Shuffled")
            st.plotly_chart(fig_imp, use_container_width=True)

    # ==========================================
    # TAB 2: DEEP VALUE METRICS
    # ==========================================
    with tab2:
        st.subheader("💎 20 Deep Value & Solvency Metrics")
        st.markdown("Statistically driven fundamental breakdown using the TTM (Trailing Twelve Months) or MRQ (Most Recent Quarter) periods.")
        
        # Calculate Graham Number
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
            ("EV to Free Cash Flow", safe_get(info, 'enterpriseValue') / safe_get(info, 'freeCashflow', default=1) if safe_get(info, 'freeCashflow') not in [None, "N/A"] else "N/A", r"\frac{\text{Enterprise Value}}{\text{Free Cash Flow}}", "TTM"),
            ("Dividend Yield", safe_get(info, 'dividendYield', format_type='pct'), r"\frac{\text{Annual Dividends per Share}}{\text{Price per Share}}", "Forward"),
            ("Payout Ratio", safe_get(info, 'payoutRatio', format_type='pct'), r"\frac{\text{Dividends Paid}}{\text{Net Income}}", "TTM"),
            ("Free Cash Flow Yield", safe_get(info, 'freeCashflow') / safe_get(info, 'marketCap', default=1) if safe_get(info, 'freeCashflow') not in [None, "N/A"] else "N/A", r"\frac{\text{Free Cash Flow}}{\text{Market Cap}}", "TTM"),
            ("Current Ratio", safe_get(info, 'currentRatio', format_type='float'), r"\frac{\text{Current Assets}}{\text{Current Liabilities}}", "MRQ"),
            ("Quick Ratio", safe_get(info, 'quickRatio', format_type='float'), r"\frac{\text{Current Assets} - \text{Inventory}}{\text{Current Liabilities}}", "MRQ"),
            ("Cash Ratio", safe_get(info, 'totalCash') / safe_get(info, 'totalDebt', default=1) if safe_get(info, 'totalCash') not in [None, "N/A"] else "N/A", r"\frac{\text{Cash \& Equivalents}}{\text{Current Liabilities}}", "MRQ"),
            ("Debt to Equity", safe_get(info, 'debtToEquity', format_type='float'), r"\frac{\text{Total Liabilities}}{\text{Shareholders' Equity}}", "MRQ"),
            ("Return on Equity (ROE)", safe_get(info, 'returnOnEquity', format_type='pct'), r"\frac{\text{Net Income}}{\text{Shareholders' Equity}}", "TTM"),
            ("Return on Assets (ROA)", safe_get(info, 'returnOnAssets', format_type='pct'), r"\frac{\text{Net Income}}{\text{Total Assets}}", "TTM"),
            ("Gross Margin", safe_get(info, 'grossMargins', format_type='pct'), r"\frac{\text{Revenue} - \text{COGS}}{\text{Revenue}}", "TTM"),
            ("Operating Margin", safe_get(info, 'operatingMargins', format_type='pct'), r"\frac{\text{Operating Income}}{\text{Revenue}}", "TTM"),
        ]

        formatted_metrics = []
        for name, val, formula, period in metrics_data:
            if isinstance(val, (float, np.float64)) and name in ["EV to Free Cash Flow", "Cash Ratio"]:
                val = f"{val:.2f}"
            elif isinstance(val, (float, np.float64)) and name in ["Free Cash Flow Yield"]:
                val = f"{val*100:.2f}%"
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
        st.markdown("Breaking down Return on Equity (ROE) to identify the true driver of returns: Profitability, Efficiency, or Leverage.")
        
        st.latex(r"ROE = \left( \frac{\text{Net Income}}{\text{Sales}} \right) \times \left( \frac{\text{Sales}}{\text{Total Assets}} \right) \times \left( \frac{\text{Total Assets}}{\text{Total Equity}} \right)")
        st.latex(r"ROE = \text{Net Profit Margin} \times \text{Asset Turnover} \times \text{Equity Multiplier}")
        
        try:
            # Safely extract latest annual financials if available
            net_income = ic.loc['Net Income'].iloc[0] if 'Net Income' in ic.index else None
            revenue = ic.loc['Total Revenue'].iloc[0] if 'Total Revenue' in ic.index else None
            total_assets = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else None
            total_equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else None
            
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
                
                st.info("💡 **Interpretation:** High ROE driven by Net Profit Margin is highly desirable (pricing power). If driven mostly by the Equity Multiplier, the company is relying heavily on debt to generate returns, indicating higher solvency risk.")
            else:
                st.warning("Insufficient balance sheet or income statement data available from standard API for a full manual DuPont calculation.")
        except Exception as e:
            st.error("Error calculating manual DuPont breakdown. Financial statements may be missing standard line items for this ticker.")
else:
    st.error("Ticker not found. Please check the symbol.")
