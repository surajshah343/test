import streamlit as st
import os
import json
from datetime import date
import yfinance as yf
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2015-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")

os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Pro Dashboard", layout="wide")

t1, t2 = st.columns([0.9, 0.1])
with t1:
    st.title('🧠 Continuous Learning AI Dashboard by S. Shah')
with t2:
    with st.popover("?"):
        st.markdown("**Dashboard Guide:** This tool uses XGBoost to predict the 'Alpha' (residual) of a stock over its rolling drift, combined with Monte Carlo simulations for risk assessment.")

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 0, 50, value=20)

show_paths = st.sidebar.toggle("Show Individual Sim Paths", value=True)

MODEL_FILE = f"saved_models/{selected_stock}_continuous_model.json"
META_FILE = f"saved_models/{selected_stock}_meta.json"

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date): # Add start_date here
    try:
        data = yf.download(ticker, start_date, TODAY)
        if data is None or data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns]
        data.reset_index(inplace=True)
        if 'Date' not in data.columns and 'index' in data.columns: data.rename(columns={'index': 'Date'}, inplace=True)
        data.dropna(subset=['Close'], inplace=True)
        return data[['Date', 'Close']].copy()
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

# When calling it:
data = load_data(selected_stock, START)

if data is None or data.empty:
    st.error(f"Error: Could not pull data for '{selected_stock}'. Check ticker symbol.")
    st.stop()

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING & TECHNICAL INDICATORS
# -----------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    delta = df['Close'].diff()
    gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    df['Avg_Gain'] = gain.ewm(alpha=1/14, adjust=False).mean()
    df['Avg_Loss'] = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = df['Avg_Gain'] / df['Avg_Loss'].replace(0, np.nan)
    df['RSI'] = np.where(df['Avg_Loss'] == 0, 100, 100 - (100 / (1 + rs)))
    
    df['Lag_1_Ret'] = df['Close'] / df['Close'].shift(1) - 1
    df['Lag_2_Ret'] = df['Close'] / df['Close'].shift(2) - 1
    df['SMA_10_Pct'] = df['SMA_10'] / df['Close'] - 1
    df['SMA_20_Pct'] = df['SMA_20'] / df['Close'] - 1
    df['MACD_Pct'] = df['MACD'] / df['Close']
    df['Vol_20'] = df['Lag_1_Ret'].rolling(window=20).std()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Rolling_Drift'] = df['Daily_Return'].rolling(window=50).mean()
    df['Target_Return'] = df['Daily_Return'].shift(-1)
    df['Target_Residual'] = df['Target_Return'] - df['Rolling_Drift']
    return df

all_data_engineered = engineer_features(data)
features = ['Lag_1_Ret', 'Lag_2_Ret', 'SMA_10_Pct', 'SMA_20_Pct', 'MACD_Pct', 'RSI', 'Vol_20', 'DayOfYear', 'Month']
target = 'Target_Residual'
full_ml_data = all_data_engineered.dropna(subset=features + [target]).copy()

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
final_model = xgb.XGBRegressor()
if not os.path.exists(MODEL_FILE):
    tuner = RandomizedSearchCV(xgb.XGBRegressor(random_state=42), {'max_depth': [3, 5], 'n_estimators': [100]}, n_iter=2, cv=2)
    tuner.fit(full_ml_data[features], full_ml_data[target])
    final_model = tuner.best_estimator_
    final_model.save_model(MODEL_FILE)
    with open(META_FILE, 'w') as f: json.dump({"last_trained_date": full_ml_data['Date'].max().strftime("%Y-%m-%d")}, f)
else:
    final_model.load_model(MODEL_FILE)

# -----------------------------------------------------------------------------
# CACHED FORECAST GENERATION (With Progress Bar)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_forecast_cached(_model, _historical_df, _dates, _num_sims):
    hist_eng = engineer_features(_historical_df)
    records = hist_eng.tail(30).to_dict('records')
    returns_tail = _historical_df['Close'].pct_change().dropna().tail(504)
    mu, sigma = returns_tail.mean(), returns_tail.std()
    daily_drift = mu - (0.5 * sigma**2)
    
    mc_paths = {f'Sim_{i}': [] for i in range(_num_sims)}
    current_mc_prices = {f'Sim_{i}': records[-1]['Close'] for i in range(_num_sims)}
    l_ema12, l_ema26 = records[-1]['EMA_12'], records[-1]['EMA_26']
    l_gain, l_loss = records[-1]['Avg_Gain'], records[-1]['Avg_Loss']
    
    preds = []
    progress_bar = st.progress(0, text="Initializing AI Forecast...")
    
    for i, date_val in enumerate(_dates):
        c_price = records[-1]['Close']
        l_ema12 = (c_price - l_ema12) * (2/13) + l_ema12
        l_ema26 = (c_price - l_ema26) * (2/27) + l_ema26
        macd = l_ema12 - l_ema26
        diff = c_price - records[-2]['Close']
        g, l = (diff, 0) if diff > 0 else (0, -diff)
        l_gain, l_loss = g * (1/14) + l_gain * (13/14), l * (1/14) + l_loss * (13/14)
        rsi = 100 if l_loss == 0 else 100 - (100 / (1 + (l_gain/l_loss)))
        vol_20 = np.std([records[x]['Close']/records[x-1]['Close']-1 for x in range(-20, 0)], ddof=1)
        
        X = pd.DataFrame({
            'Lag_1_Ret': [c_price/records[-2]['Close']-1], 'Lag_2_Ret': [c_price/records[-3]['Close']-1],
            'SMA_10_Pct': [(np.mean([r['Close'] for r in records[-10:]])/c_price)-1],
            'SMA_20_Pct': [(np.mean([r['Close'] for r in records[-20:]])/c_price)-1],
            'MACD_Pct': [macd/c_price], 'RSI': [rsi], 'Vol_20': [vol_20],
            'DayOfYear': [date_val.dayofyear], 'Month': [date_val.month]
        })
        
        ai_alpha = _model.predict(X)[0]
        f_return = daily_drift + ai_alpha + np.random.normal(0, sigma)
        new_close = c_price * np.exp(f_return)
        
        for s in range(_num_sims):
            s_ret = daily_drift + ai_alpha + np.random.normal(0, sigma)
            new_s_p = current_mc_prices[f'Sim_{s}'] * np.exp(s_ret)
            mc_paths[f'Sim_{s}'].append(max(0.01, new_s_p))
            current_mc_prices[f'Sim_{s}'] = new_s_p
            
        u_bound, l_bound = new_close * np.exp(sigma * 1.96 * np.sqrt(i+1)), new_close * np.exp(-sigma * 1.96 * np.sqrt(i+1))
        rec = {'Date': date_val, 'Close': new_close, 'Upper_Bound': u_bound, 'Lower_Bound': l_bound}
        records.append(rec); preds.append(rec)
        if len(records) > 60: records.pop(0)
        
        if i % 10 == 0 or i == len(_dates) - 1:
            progress_bar.progress((i + 1) / len(_dates), text=f"Processing Day {i+1} of {len(_dates)}...")
            
    progress_bar.empty()
    return pd.DataFrame(preds), mc_paths

# Execution
future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
f_forecast, mc_paths = generate_forecast_cached(final_model, data, future_dates, n_simulations)

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
plot_data = engineer_features(pd.concat([data, f_forecast], ignore_index=True))
split_date_idx = int(len(full_ml_data) * 0.8)
split_date = full_ml_data.iloc[split_date_idx]['Date']

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.4, 0.2, 0.2, 0.2],
                    subplot_titles=('Institutional Forecast', 'Bollinger Bands (Price vs Volatility)', 'RSI (Momentum)', 'MACD (Trend)'))

fig.add_trace(go.Scatter(x=data[data['Date'] < split_date]['Date'], y=data[data['Date'] < split_date]['Close'], name='Train', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=data[data['Date'] >= split_date]['Date'], y=data[data['Date'] >= split_date]['Close'], name='Test', line=dict(color='limegreen')), row=1, col=1)
fig.add_trace(go.Scatter(x=f_forecast['Date'], y=f_forecast['Close'], name='AI Forecast', line=dict(color='blue', width=2.5)), row=1, col=1)

if show_paths:
    for s in mc_paths:
        fig.add_trace(go.Scatter(x=f_forecast['Date'], y=mc_paths[s], mode='lines', line=dict(color='rgba(0,150,255,0.05)'), showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=f_forecast['Date'], y=f_forecast['Upper_Bound'], line=dict(width=0), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=f_forecast['Date'], y=f_forecast['Lower_Bound'], fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', name='95% Forecast CI'), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], name='Price', line=dict(color='black', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Upper_Band'], line=dict(color='rgba(128,128,128,0.5)', dash='dot'), name='Upper BB'), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Lower_Band'], line=dict(color='rgba(128,128,128,0.5)', dash='dot'), fill='tonexty', name='Lower BB'), row=2, col=1)

fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['MACD'], line=dict(color='blue'), name='MACD Line'), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Signal_Line'], line=dict(color='red', dash='dot'), name='Signal Line'), row=4, col=1)
colors = ['green' if val >= 0 else 'red' for val in plot_data['MACD_Hist']]
fig.add_trace(go.Bar(x=plot_data['Date'], y=plot_data['MACD_Hist'], marker_color=colors, name='MACD Histogram'), row=4, col=1)

for row, txt in [(1, "Forecast Main"), (2, "Bollinger Bands"), (3, "RSI Momentum"), (4, "MACD Trend")]:
    fig.add_annotation(x=0.01, y=0.95, xref=f"x{row if row > 1 else ''} domain", yref=f"y{row if row > 1 else ''} domain",
                       text="<b>?</b>", showarrow=False, bgcolor="black", font=dict(color="white"), hovertext=txt)

fig.update_yaxes(range=[0, data['Close'].iloc[-1] * 3], row=1, col=1)
fig.update_layout(height=1100, template='plotly_white', hovermode='x unified', legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# PRICE TARGET SUMMARY TABLE
# -----------------------------------------------------------------------------
st.markdown("### 🎯 Institutional Price Targets")
cur_price = data['Close'].iloc[-1]

# Extract targets for key milestones
targets = []
milestones = {
    "6 Months": 126,
    "1 Year": 252,
    "2 Years": 504,
    "4 Years": 1008
}

for label, day_idx in milestones.items():
    if day_idx <= len(f_forecast):
        row = f_forecast.iloc[day_idx - 1]
        proj_p = row['Close']
        roi = ((proj_p / cur_price) - 1) * 100
        targets.append({
            "Horizon": label,
            "Target Date": row['Date'].strftime('%Y-%m-%d'),
            "Projected Price": f"${proj_p:.2f}",
            "Lower Bound (95%)": f"${row['Lower_Bound']:.2f}",
            "Upper Bound (95%)": f"${row['Upper_Bound']:.2f}",
            "Potential ROI": f"{roi:+.2f}%"
        })

if targets:
    st.table(pd.DataFrame(targets))
