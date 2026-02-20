import streamlit as st
import os
import json
from datetime import date, datetime
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# --- CONFIGURATION ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Pro Dashboard v3.2", layout="wide")
st.title('🧠 Dynamic Financial AI: Regime-Aware Framework')

# --- SIDEBAR & TICKER STATE ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="NVDA").upper()

# Force clear cache if ticker changes
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = ticker_input
if st.session_state.current_ticker != ticker_input:
    st.cache_data.clear()
    st.session_state.current_ticker = ticker_input

n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)

# RETRAIN TRIGGER
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

MODEL_FILE = f"saved_models/{ticker_input}_v3.json"
META_FILE = f"saved_models/{ticker_input}_meta_v3.json"

# --- 1. DYNAMIC DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, end=TODAY)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.dropna().copy()
    except Exception as e:
        st.error(f"Download Error for {ticker}: {e}")
        return None

data = load_data(ticker_input, START)
if data is None: st.stop()

# --- 2. DYNAMIC FEATURE ENGINEERING ---
def engineer_features(df):
    df = df.copy()
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['Close'].rolling(20).mean() / df['Close']) - 1
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    df['Drift_50'] = df['Log_Ret'].rolling(50).mean()
    df['Target_Residual'] = df['Log_Ret'].shift(-1) - df['Drift_50']
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    df['Month'] = df['Date'].dt.month / 12.0
    return df.dropna().copy()

ml_data = engineer_features(data)
features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear', 'Month']
target = 'Target_Residual'
X, y = ml_data[features], ml_data[target]

# --- 3. TICKER-SPECIFIC TRAINING ---
def train_ticker_model(X, y, ticker):
    with st.status(f"Training specialized model for {ticker}...", expanded=True) as status:
        tscv = TimeSeriesSplit(n_splits=5) 
        tuner = RandomizedSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            param_distributions={'max_depth': [3, 6, 9], 'n_estimators': [100, 200]},
            cv=tscv, n_iter=5, n_jobs=-1
        )
        tuner.fit(X, y)
        status.update(label="Training Complete!", state="complete", expanded=False)
    return tuner.best_estimator_

# Check for existing model OR retrain command
if not os.path.exists(MODEL_FILE) or retrain_button:
    final_model = train_ticker_model(X, y, ticker_input)
    final_model.save_model(MODEL_FILE)
    with open(META_FILE, 'w') as f:
        json.dump({"ticker": ticker_input, "last_trained": ml_data['Date'].max().strftime("%Y-%m-%d")}, f)
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)
    st.sidebar.caption(f"Using cached model for {ticker_input}")

# --- 4. STOCHASTIC SIMULATION ---
@st.cache_data(show_spinner=False)
def run_simulation(_model, _historical_df, _n_days, _n_sims, _ticker):
    last_price = _historical_df['Close'].iloc[-1]
    hist_tail = _historical_df['Log_Ret'].tail(252)
    mu, sigma = hist_tail.mean(), hist_tail.std()
    daily_drift_adj = mu - 0.5 * (sigma**2)
    
    all_paths = np.zeros((_n_days, _n_sims))
    current_prices = np.full(_n_sims, last_price)
    current_log_ret = _historical_df['Log_Ret'].iloc[-1]
    
    progress_bar = st.progress(0, text=f"Simulating {_ticker} Dynamics...")
    for d in range(_n_days):
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': [current_log_ret],
            'SMA_20_Pct': [(_historical_df['Close'].tail(20).mean() / current_prices.mean()) - 1],
            'Vol_20': [sigma],
            'DayOfYear': [(datetime.now().timetuple().tm_yday + d) % 366 / 366.0],
            'Month': [(datetime.now().month + (d // 30) - 1) % 12 / 12.0]
        })
        alpha_pred = _model.predict(pred_feat)[0]
        shocks = np.random.normal(0, sigma, _n_sims)
        log_returns = daily_drift_adj + alpha_pred + shocks
        current_prices = current_prices * np.exp(log_returns)
        all_paths[d, :] = current_prices
        current_log_ret = np.mean(log_returns)
        if d % 50 == 0: progress_bar.progress((d+1)/_n_days)
    progress_bar.empty()
    return all_paths

sim_results = run_simulation(final_model, ml_data, forecast_days, n_simulations, ticker_input)

# --- 5. VISUALIZATION ---
st.subheader(f"📊 Forecast Insights: {ticker_input}")
median_p = np.median(sim_results, axis=1)
upper_ci = np.percentile(sim_results, 97.5, axis=1)
lower_ci = np.percentile(sim_results, 2.5, axis=1)

# Backtest Performance for UI
split = int(len(ml_data) * 0.8)
test_preds = final_model.predict(ml_data.iloc[split:][features])
rmse = np.sqrt(mean_squared_error(ml_data.iloc[split:][target], test_preds))

c1, c2, c3 = st.columns(3)
c1.metric("Model RMSE (Alpha)", f"{rmse:.5f}")
c2.metric("Regime Volatility (σ)", f"{ml_data['Log_Ret'].tail(20).std():.4f}")
c3.metric("Last Close", f"${ml_data['Close'].iloc[-1]:.2f}")



fig = go.Figure()
fig.add_trace(go.Scatter(x=ml_data['Date'].tail(252), y=ml_data['Close'].tail(252), name='Historical', line=dict(color='#2c3e50')))
future_dates = pd.date_range(ml_data['Date'].max(), periods=forecast_days+1, freq='B')[1:]
fig.add_trace(go.Scatter(x=future_dates, y=upper_ci, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=lower_ci, line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.15)', name='95% Confidence Interval'))
fig.add_trace(go.Scatter(x=future_dates, y=median_p, name='AI Median Path', line=dict(color='#3498db', width=3)))
fig.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Horizons Table
horizons = {"6 Months": 126, "1 Year": 252}
summary_list = []
for label, idx in horizons.items():
    if idx <= forecast_days:
        prices = sim_results[idx-1, :]
        summary_list.append({
            "Horizon": label,
            "Median Target": f"${np.median(prices):.2f}",
            "95% Confidence Range": f"${np.percentile(prices, 2.5):.2f} - ${np.percentile(prices, 97.5):.2f}",
            "Prob. of Growth": f"{(prices > ml_data['Close'].iloc[-1]).mean()*100:.1f}%"
        })
st.table(pd.DataFrame(summary_list))
