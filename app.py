import streamlit as st
import os
import json
from datetime import date
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2015-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Pro Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()
n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 0, 50, value=20)
show_paths = st.sidebar.toggle("Show Individual Sim Paths", value=True)

MODEL_FILE = f"saved_models/{selected_stock}_continuous_model.json"

# -----------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    try:
        data = yf.download(ticker, start_date, TODAY)
        if data is None or data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns]
        data.reset_index(inplace=True)
        data.dropna(subset=['Close'], inplace=True)
        return data[['Date', 'Close']].copy()
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

def engineer_features(df):
    df = df.copy()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    
    # Technical Indicators
    df['SMA_10_Pct'] = (df['Close'].rolling(10).mean() / df['Close']) - 1
    df['SMA_20_Pct'] = (df['Close'].rolling(20).mean() / df['Close']) - 1
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (delta.where(delta < 0, 0).abs()).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Lag_1_Ret'] = df['Close'].pct_change()
    df['Vol_20'] = df['Lag_1_Ret'].rolling(window=20).std()
    df['Rolling_Drift'] = df['Lag_1_Ret'].rolling(window=50).mean()
    
    # Targets
    df['Target_Return'] = df['Lag_1_Ret'].shift(-1)
    df['Target_Residual'] = df['Target_Return'] - df['Rolling_Drift']
    return df

data = load_data(selected_stock, START)
if data is None: st.stop()

all_data_engineered = engineer_features(data)
features = ['Lag_1_Ret', 'SMA_10_Pct', 'SMA_20_Pct', 'RSI', 'Vol_20', 'DayOfYear', 'Month']
target = 'Target_Residual'
full_ml_data = all_data_engineered.dropna(subset=features + [target]).copy()

# -----------------------------------------------------------------------------
# BACKTESTING & ACCURACY EVALUATION
# -----------------------------------------------------------------------------
# Split data: 80% for training, 20% for testing (Out-of-sample)
split_idx = int(len(full_ml_data) * 0.8)
train_df = full_ml_data.iloc[:split_idx]
test_df = full_ml_data.iloc[split_idx:]

# Model Training
model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(train_df[features], train_df[target])

# Backtest Predictions
test_preds_residual = model.predict(test_df[features])

# 1. Directional Accuracy (Hit Ratio)
# Did the AI correctly predict if the stock would outperform or underperform its drift?
actual_dir = np.sign(test_df[target])
pred_dir = np.sign(test_preds_residual)
directional_accuracy = (actual_dir == pred_dir).mean() * 100

# 2. Price-Based MAPE (Correcting the "High MAPE" issue)
# We convert predicted residuals back into predicted prices to get a real error %
prev_close = data.iloc[test_df.index - 1]['Close'].values
actual_close = test_df['Close'].values
# Pred Price = Prev Close * (1 + Drift + Predicted Residual)
pred_close = prev_close * (1 + test_df['Rolling_Drift'] + test_preds_residual)
price_mape = mean_absolute_percentage_error(actual_close, pred_close) * 100

# Display Metrics
st.title(f"📈 AI Analysis: {selected_stock}")
m1, m2, m3 = st.columns(3)
m1.metric("Directional Accuracy", f"{directional_accuracy:.2f}%", help="Percentage of days the AI correctly predicted the 'Alpha' direction.")
m2.metric("Price Accuracy (MAPE)", f"{100 - price_mape:.2f}%", help="How close the AI predicted price was to the actual closing price.")
m3.metric("Backtest Period", f"{len(test_df)} Days")

# -----------------------------------------------------------------------------
# FORECASTING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_forecast(_model, _historical_df, _dates, _num_sims):
    hist_eng = engineer_features(_historical_df)
    records = hist_eng.tail(60).to_dict('records')
    
    returns_tail = _historical_df['Close'].pct_change().dropna().tail(252)
    mu, sigma = returns_tail.mean(), returns_tail.std()
    
    preds = []
    mc_paths = {f'Sim_{i}': [] for i in range(_num_sims)}
    current_mc_prices = {f'Sim_{i}': records[-1]['Close'] for i in range(_num_sims)}

    for i, date_val in enumerate(_dates):
        c_price = records[-1]['Close']
        
        # Simple Feature updates for iteration
        X = pd.DataFrame({
            'Lag_1_Ret': [records[-1]['Target_Return'] if 'Target_Return' in records[-1] else 0],
            'SMA_10_Pct': [(np.mean([r['Close'] for r in records[-10:]])/c_price)-1],
            'SMA_20_Pct': [(np.mean([r['Close'] for r in records[-20:]])/c_price)-1],
            'RSI': [records[-1]['RSI']],
            'Vol_20': [sigma],
            'DayOfYear': [date_val.dayofyear],
            'Month': [date_val.month]
        })
        
        ai_alpha = _model.predict(X)[0]
        drift = mu 
        
        # Main Forecast Path
        f_return = drift + ai_alpha
        new_close = c_price * (1 + f_return)
        
        # Monte Carlo Paths
        for s in range(_num_sims):
            s_ret = drift + ai_alpha + np.random.normal(0, sigma)
            new_s_p = current_mc_prices[f'Sim_{s}'] * (1 + s_ret)
            mc_paths[f'Sim_{s}'].append(max(0.01, new_s_p))
            current_mc_prices[f'Sim_{s}'] = new_s_p
            
        u_bound = new_close * (1 + (sigma * 1.96 * np.sqrt(i+1)))
        l_bound = new_close * (1 - (sigma * 1.96 * np.sqrt(i+1)))
        
        rec = {'Date': date_val, 'Close': new_close, 'Upper_Bound': u_bound, 'Lower_Bound': l_bound, 'RSI': records[-1]['RSI']}
        records.append(rec); preds.append(rec)
        if len(records) > 60: records.pop(0)
        
    return pd.DataFrame(preds), mc_paths

future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
f_forecast, mc_paths = generate_forecast(model, data, future_dates, n_simulations)

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(x=data['Date'].iloc[-150:], y=data['Close'].iloc[-150:], name='Historical Price', line=dict(color='black')))

# Forecast
fig.add_trace(go.Scatter(x=f_forecast['Date'], y=f_forecast['Close'], name='AI Mean Forecast', line=dict(color='blue', width=3)))

# Confidence Intervals
fig.add_trace(go.Scatter(x=f_forecast['Date'], y=f_forecast['Upper_Bound'], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=f_forecast['Date'], y=f_forecast['Lower_Bound'], fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='95% Confidence Interval'))

if show_paths:
    for s in mc_paths:
        fig.add_trace(go.Scatter(x=f_forecast['Date'], y=mc_paths[s], mode='lines', line=dict(color='rgba(0,150,255,0.05)'), showlegend=False))

fig.update_layout(title=f"{selected_stock} AI Forecast ({n_years} Year)", template='plotly_white', height=600)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TARGET TABLE
# -----------------------------------------------------------------------------
st.subheader("🎯 Price Targets")
milestones = {"6 Months": 126, "1 Year": 252, "2 Years": 504}
target_rows = []
for label, idx in milestones.items():
    if idx < len(f_forecast):
        row = f_forecast.iloc[idx]
        target_rows.append({
            "Horizon": label,
            "Date": row['Date'].strftime('%Y-%m-%d'),
            "Projected": f"${row['Close']:.2f}",
            "Range": f"${row['Lower_Bound']:.2f} - ${row['Upper_Bound']:.2f}"
        })
st.table(pd.DataFrame(target_rows))
