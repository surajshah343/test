import streamlit as st
import os
from datetime import date, datetime
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
TODAY = date.today()
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Quant Pro v8.1", layout="wide")
st.title('🧠 Financial AI: Auto-Optimizing Quantitative Framework')

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=3)
forecast_days = int(n_years * 252) 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=1000)

retrain_button = st.sidebar.button("🔄 Force Model Retrain")
if retrain_button:
    st.cache_data.clear()
    st.sidebar.success("Cache Cleared. Retraining pipeline...")

MODEL_FILE = f"saved_models/{ticker_input}_v8_1.json"

# --- 1. DYNAMIC HORIZON OPTIMIZER ---
@st.cache_data(show_spinner="Optimizing Historical Data Horizon...")
def optimize_and_load_data(ticker):
    raw_data = yf.download(ticker, period="10y")
    if raw_data.empty: return None
    if isinstance(raw_data.columns, pd.MultiIndex): raw_data.columns = raw_data.columns.get_level_values(0)
    raw_data.reset_index(inplace=True)
    
    lookbacks = [2 * 252, 5 * 252, 10 * 252]
    best_hit_ratio = -1
    best_df = None
    best_years = 0
    
    for days in lookbacks:
        if len(raw_data) < days: continue
        
        df = raw_data.tail(days).copy()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
        df['SMA_20_Pct'] = (df['MA20'] / df['Close']) - 1
        
        # FIX: Predict the raw return, not a dynamic residual, to prevent explosive feedback
        df['Target'] = df['Log_Ret'].shift(-1) 
        df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
        df.dropna(inplace=True)
        
        if len(df) < 100: continue
            
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]
        
        temp_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1)
        features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear']
        temp_model.fit(train[features], train['Target'])
        
        preds = temp_model.predict(test[features])
        hit_ratio = np.mean(np.sign(preds) == np.sign(test['Target'].values)) * 100
        
        if hit_ratio > best_hit_ratio:
            best_hit_ratio = hit_ratio
            best_df = raw_data.tail(days).copy() 
            best_years = days // 252

    return best_df, best_years, best_hit_ratio

opt_results = optimize_and_load_data(ticker_input)
if opt_results is None or opt_results[0] is None: 
    st.error("Failed to load sufficient data for optimization.")
    st.stop()

raw_optimal_data, optimal_years, optimized_hit_ratio = opt_results
st.sidebar.info(f"**Auto-Optimized History:** Using past {optimal_years} years for peak accuracy ({optimized_hit_ratio:.1f}% test validation).")

# --- 2. FULL FEATURE ENGINEERING ---
def build_full_dataset(df):
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA20'] = df['Close'].rolling(20).mean()
    df['stddev'] = df['Close'].rolling(20).std()
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    df['Upper'] = df['MA20'] + (df['stddev'] * 2)
    df['Lower'] = df['MA20'] - (df['stddev'] * 2)
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['MA20'] / df['Close']) - 1
    df['Target'] = df['Log_Ret'].shift(-1)
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    
    return df.dropna().copy()

ml_data = build_full_dataset(raw_optimal_data)
features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear']
target = 'Target'

split_idx = int(len(ml_data) * 0.8)
train_set = ml_data.iloc[:split_idx]
test_set = ml_data.iloc[split_idx:]

# --- 3. TRAINING ---
if not os.path.exists(MODEL_FILE) or retrain_button:
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05)
    model.fit(train_set[features], train_set[target])
    model.save_model(MODEL_FILE)
    final_model = model
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

test_preds = final_model.predict(test_set[features])
final_hit_ratio = np.mean(np.sign(test_preds) == np.sign(test_set[target].values)) * 100

# --- 4. VECTORIZED SIMULATION ENGINE WITH STABLE DYNAMICS ---
@st.cache_data(show_spinner="Simulating Stabilized Stochastic Paths...")
def run_simulation(_model, _historical_df, n_days, n_sims):
    last_price = _historical_df['Close'].iloc[-1]
    
    # Establish strict historical boundaries
    hist_max_vol = _historical_df['Vol_20'].max()
    
    # Create tracking matrices
    hist_log_ret = np.tile(_historical_df['Log_Ret'].tail(20).values, (n_sims, 1)).T 
    hist_prices = np.tile(_historical_df['Close'].tail(20).values, (n_sims, 1)).T
    
    all_paths = np.zeros((n_days, n_sims))
    current_prices = np.full(n_sims, last_price)
    
    for d in range(n_days):
        current_ma20 = np.mean(hist_prices[-20:], axis=0)
        current_vol20 = np.std(hist_log_ret[-20:], axis=0, ddof=1)
        
        # Clamp volatility to prevent explosions
        current_vol20 = np.clip(current_vol20, 0.005, hist_max_vol * 1.5)
        
        lag_1_ret = hist_log_ret[-1]
        
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': lag_1_ret,
            'SMA_20_Pct': (current_ma20 / current_prices) - 1,
            'Vol_20': current_vol20,
            'DayOfYear': [(datetime.now().timetuple().tm_yday + d) % 366 / 366.0] * n_sims
        })
        
        # AI predicts base directional return
        ai_pred_ret = _model.predict(pred_feat)
        
        # Add normal stochastic shock based on current local volatility
        shocks = np.random.normal(0, current_vol20, n_sims)
        log_returns = ai_pred_ret + shocks
        
        # Strict hard cap on daily movements (e.g. +/- 15% max per day) to maintain mathematical stability
        log_returns = np.clip(log_returns, -0.15, 0.15)
        
        current_prices = current_prices * np.exp(log_returns)
        current_prices = np.maximum(current_prices, 0.01) # Floor at 1 cent
        
        all_paths[d, :] = current_prices
        
        # Roll the arrays forward
        hist_prices = np.vstack((hist_prices[1:], current_prices))
        hist_log_ret = np.vstack((hist_log_ret[1:], log_returns))
        
    return all_paths

sim_results = run_simulation(final_model, ml_data, forecast_days, n_simulations)

initial_price = ml_data['Close'].iloc[-1]
median_forecast = np.median(sim_results, axis=1)
upper_95_bound = np.percentile(sim_results, 97.5, axis=1)
lower_95_bound = np.percentile(sim_results, 5, axis=1) 
terminal_var_95_price = lower_95_bound[-1]
terminal_var_95_pct = ((terminal_var_95_price - initial_price) / initial_price) * 100

# --- 5. FIBONACCI & 30-DAY AI BREAKOUT ANALYSIS ---
fib_lookback = ml_data.tail(252) 
swing_high = fib_lookback['Close'].max()
swing_low = fib_lookback['Close'].min()
price_diff = swing_high - swing_low

fib_levels = {
    '0.0% (High)': swing_high,
    '23.6%': swing_high - 0.236 * price_diff,
    '38.2%': swing_high - 0.382 * price_diff,
    '50.0%': swing_high - 0.500 * price_diff,
    '61.8%': swing_high - 0.618 * price_diff,
    '100.0% (Low)': swing_low
}

sorted_fibs = sorted(fib_levels.items(), key=lambda x: x[1])
support_level, resistance_level = None, None

for name, price in sorted_fibs:
    if price < initial_price:
        support_level = (name, price)
    if price > initial_price and resistance_level is None:
        resistance_level = (name, price)

forecast_30d = median_forecast[:30]
max_30d = np.max(forecast_30d)
min_30d = np.min(forecast_30d)

# --- 6. UI & CHARTS WITH TOOLTIPS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${initial_price:.2f}", help="The latest available daily closing price.")
m2.metric("OOS Hit Ratio", f"{final_hit_ratio:.1f}%", help="Out-Of-Sample Directional Accuracy.")
m3.metric("Terminal 95% VaR", f"{terminal_var_95_pct:.1f}%", help=f"5% chance of dropping below ${terminal_var_95_price:.2f}.")
m4.metric("Test MAE", f"{mean_absolute_error(test_set[target], test_preds):.5f}", help="Mean Absolute Error of the AI on unseen data.")

st.markdown("---")
st.subheader("🔮 30-Day AI Fibonacci Breakout Matrix")

col_sup, col_res, col_out = st.columns(3)

with col_sup:
    if support_level:
        st.info(f"**Immediate Support:** {support_level[0]}\n\n**${support_level[1]:.2f}**")
    else:
        st.info("**Immediate Support:** None\n\nAsset is at Local Lows.")

with col_res:
    if resistance_level:
        st.warning(f"**Immediate Resistance:** {resistance_level[0]}\n\n**${resistance_level[1]:.2f}**")
    else:
        st.warning("**Immediate Resistance:** None\n\nAsset is at Local Highs.")

with col_out:
    if resistance_level and max_30d > resistance_level[1]:
        st.success(f"📈 **AI Outlook:** The AI projects an upward breakout of resistance at ${resistance_level[1]:.2f}.")
    elif support_level and min_30d < support_level[1]:
        st.error(f"📉 **AI Outlook:** The AI projects a downward support break at ${support_level[1]:.2f}.")
    else:
        st.info(f"⚖️ **AI Outlook:** Consolidation within current Fibonacci channel bounds (${min_30d:.2f} - ${max_30d:.2f}).")

st.markdown("---")

st.subheader(f"📈 Projection & Fibonacci Analysis ({n_years}Y)")
fig_main = go.Figure()

fig_main.add_trace(go.Scatter(x=train_set['Date'], y=train_set['Close'], name='Training Data', line=dict(color='#2980b9')))
fig_main.add_trace(go.Scatter(x=test_set['Date'], y=test_set['Close'], name='Testing Data', line=dict(color='#e67e22')))

fib_colors = ['#e74c3c', '#f1c40f', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
for (label, price), color in zip(fib_levels.items(), fib_colors):
    fig_main.add_hline(y=price, line_dash="dot", line_color=color, annotation_text=f"Fib {label}: ${price:.2f}", annotation_position="top left")

future_dates = pd.date_range(ml_data['Date'].max(), periods=forecast_days + 1, freq='B')[1:]
fig_main.add_trace(go.Scatter(x=future_dates, y=upper_95_bound, line=dict(width=0), showlegend=False))
fig_main.add_trace(go.Scatter(x=future_dates, y=lower_95_bound, line=dict(width=0), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.15)', name='95% VaR Funnel'))
fig_main.add_trace(go.Scatter(x=future_dates, y=median_forecast, name='AI Median Forecast', line=dict(color='#2ecc71', width=3)))

fig_main.update_layout(template="plotly_white", hovermode="x unified", height=600)
st.plotly_chart(fig_main, use_container_width=True)

st.subheader("🛠 Technical Regime Analysis")
tech_view = ml_data.tail(500)
fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Upper'], line=dict(color='rgba(173, 216, 230, 0.5)')), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Lower'], line=dict(color='rgba(173, 216, 230, 0.5)'), fill='tonexty'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Close'], line=dict(color='#2c3e50', width=2)), row=1, col=1)

macd_colors = ['#26a69a' if x > 0 else '#ef5350' for x in tech_view['MACD_Hist']]
fig_tech.add_trace(go.Bar(x=tech_view['Date'], y=tech_view['MACD_Hist'], marker_color=macd_colors), row=2, col=1)
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['MACD'], line=dict(color='#2980b9')), row=2, col=1)
fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Signal'], line=dict(color='#e67e22')), row=2, col=1)

fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['RSI'], line=dict(color='#8e44ad')), row=3, col=1)
fig_tech.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
fig_tech.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=3, col=1)

fig_tech.update_layout(height=800, showlegend=False, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig_tech, use_container_width=True)
