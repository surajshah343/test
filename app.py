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

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "1995-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")

os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="Pro Dashboard", layout="wide")
st.title('🧠 Continuous Learning AI Dashboard by S. Shah')

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 0, 50, value=20)

MODEL_FILE = f"saved_models/{selected_stock}_continuous_model.json"
META_FILE = f"saved_models/{selected_stock}_meta.json"

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker):
    try:
        clean_ticker = str(ticker).split('-')[0].split(' ')[0].strip().upper()
        data = yf.download(clean_ticker, START, TODAY)
        
        if data is None or data.empty:
            return None
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data.reset_index(inplace=True)
        if 'Date' not in data.columns and 'index' in data.columns:
            data.rename(columns={'index': 'Date'}, inplace=True)
            
        data.dropna(subset=['Close'], inplace=True)
        return data[['Date', 'Close']].copy()
    except Exception:
        return None

data = load_data(selected_stock)

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
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
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
# OUT-OF-SAMPLE EVALUATION (CHRONOLOGICAL SPLIT)
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance (Holdout Set)")

# 80/20 chronological split to test how the model behaves on unseen recent data
split_idx = int(len(full_ml_data) * 0.8)
train_eval = full_ml_data.iloc[:split_idx]
test_eval = full_ml_data.iloc[split_idx:]

eval_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
eval_model.fit(train_eval[features], train_eval[target])

predictions = eval_model.predict(test_eval[features])
actuals = test_eval[target]

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
# To avoid MAPE explosion when actuals are near 0, add a tiny epsilon
mape = mean_absolute_percentage_error(actuals + 1e-8, predictions + 1e-8)

col1, col2 = st.sidebar.columns(2)
col1.metric("MAE (Residual)", f"{mae:.4f}")
col2.metric("RMSE", f"{rmse:.4f}")
st.sidebar.metric("MAPE (Residual Volatility)", f"{mape:.2%}")
st.sidebar.caption("Evaluated chronologically on the most recent 20% of historical data.")

# -----------------------------------------------------------------------------
# CONTINUOUS LEARNING ENGINE (TRAINING ON ALL DATA FOR FUTURE FORECAST)
# -----------------------------------------------------------------------------
final_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
is_new_model = not os.path.exists(MODEL_FILE)

trainable_data = full_ml_data.copy() 
latest_available_date = trainable_data['Date'].max().strftime("%Y-%m-%d")

if is_new_model:
    st.sidebar.warning("⚠️ No saved brain found. Training baseline model...")
    with st.spinner("Compiling initial AI brain..."):
        final_model.fit(trainable_data[features], trainable_data[target])
        final_model.save_model(MODEL_FILE)
        
        with open(META_FILE, 'w') as f:
            json.dump({"last_trained_date": latest_available_date}, f)
else:
    final_model.load_model(MODEL_FILE)
    
    last_trained_date = "1990-01-01"
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            meta = json.load(f)
            last_trained_date = meta.get("last_trained_date", "1990-01-01")
    
    if latest_available_date > last_trained_date:
        new_data = trainable_data[trainable_data['Date'] > last_trained_date]
        if not new_data.empty:
            with st.spinner("Learning from recent market data..."):
                final_model.fit(new_data[features], new_data[target], xgb_model=MODEL_FILE)
                final_model.save_model(MODEL_FILE)
                with open(META_FILE, 'w') as f:
                    json.dump({"last_trained_date": latest_available_date}, f)

# -----------------------------------------------------------------------------
# INSTITUTIONAL AUTOREGRESSIVE FORECAST (GBM + ML RESIDUALS)
# -----------------------------------------------------------------------------
def generate_autoregressive_forecast(trained_model, historical_df, start_buffer, dates_to_predict, num_sims):
    engineered_hist = engineer_features(historical_df)
    records = engineered_hist.tail(30).to_dict('records')
    predictions = []
    
    long_term_returns = historical_df['Close'].pct_change().dropna().tail(504) 
    mu = long_term_returns.mean()
    sigma = long_term_returns.std()
    
    daily_drift = mu - (0.5 * sigma**2)
    
    base_close = records[-1]['Close']
    mc_paths = {f'Sim_{i}': [] for i in range(num_sims)}
    current_mc_prices = {f'Sim_{i}': base_close for i in range(num_sims)}
    
    last_ema_12 = records[-1]['EMA_12']
    last_ema_26 = records[-1]['EMA_26']
    last_avg_gain = records[-1]['Avg_Gain']
    last_avg_loss = records[-1]['Avg_Loss']
    
    for i, date_val in enumerate(dates_to_predict):
        last_rec = records[-1]
        prev_rec = records[-2]
        
        current_close = last_rec['Close']
        closes = [r['Close'] for r in records[-20:]]
        
        lag_1_ret = current_close / prev_rec['Close'] - 1
        lag_2_ret = current_close / records[-3]['Close'] - 1
        
        sma_10 = np.mean([r['Close'] for r in records[-10:]])
        sma_20 = np.mean(closes)
        
        alpha_12 = 2 / (12 + 1)
        alpha_26 = 2 / (26 + 1)
        
        last_ema_12 = (current_close - last_ema_12) * alpha_12 + last_ema_12
        last_ema_26 = (current_close - last_ema_26) * alpha_26 + last_ema_26
        macd = last_ema_12 - last_ema_26
        
        delta = current_close - prev_rec['Close']
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0
        
        last_avg_gain = gain * (1/14) + last_avg_gain * (1 - 1/14)
        last_avg_loss = loss * (1/14) + last_avg_loss * (1 - 1/14)
        
        if last_avg_loss == 0:
            rsi = 100
        else:
            rs = last_avg_gain / last_avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        recent_returns = [records[idx]['Close'] / records[idx-1]['Close'] - 1 for idx in range(-20, 0)]
        vol_20 = np.std(recent_returns, ddof=1)
        
        X_pred = pd.DataFrame({
            'Lag_1_Ret': [lag_1_ret],
            'Lag_2_Ret': [lag_2_ret],
            'SMA_10_Pct': [(sma_10 / current_close) - 1],
            'SMA_20_Pct': [(sma_20 / current_close) - 1],
            'MACD_Pct': [macd / current_close if current_close != 0 else 0],
            'RSI': [rsi],
            'Vol_20': [vol_20],
            'DayOfYear': [date_val.dayofyear],
            'Month': [date_val.month]
        })
        
        ai_residual = trained_model.predict(X_pred)[0]
        baseline_noise = np.random.normal(0, sigma) 
        
        blended_return = daily_drift + ai_residual + baseline_noise
        new_close = current_close * np.exp(blended_return)
        
        for sim_idx in range(num_sims):
            noise = np.random.normal(0, sigma)
            sim_return = daily_drift + ai_residual + noise
            
            new_sim_price = current_mc_prices[f'Sim_{sim_idx}'] * np.exp(sim_return)
            new_sim_price = max(0.01, new_sim_price) 
            
            mc_paths[f'Sim_{sim_idx}'].append(new_sim_price)
            current_mc_prices[f'Sim_{sim_idx}'] = new_sim_price
            
        step_uncertainty_pct = sigma * 1.96 * np.sqrt(i + 1)
        upper_bound = new_close * np.exp(step_uncertainty_pct)
        lower_bound = max(0.01, new_close * np.exp(-step_uncertainty_pct)) 
        
        new_record = {
            'Date': date_val, 
            'Close': new_close, 
            'RSI': rsi,
            'EMA_12': last_ema_12,
            'EMA_26': last_ema_26,
            'Avg_Gain': last_avg_gain,
            'Avg_Loss': last_avg_loss,
            'Upper_Bound': upper_bound,
            'Lower_Bound': lower_bound
        }
        
        records.append(new_record)
        predictions.append(new_record)
        
        if len(records) > 50:
            records.pop(0)
            
    df_preds = pd.DataFrame(predictions)[['Date', 'Close', 'Upper_Bound', 'Lower_Bound']]
    return df_preds, mc_paths

# -----------------------------------------------------------------------------
# FUTURE FORECAST GENERATION
# -----------------------------------------------------------------------------
with st.spinner(f"Running Institutional Forecast & {n_simulations} Simulations..."):
    future_start_buffer = data[['Date', 'Close']].tail(30).copy()
    last_actual_date = future_start_buffer['Date'].iloc[-1]
    
    future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    future_forecast, mc_paths_data = generate_autoregressive_forecast(final_model, data, future_start_buffer, future_dates, n_simulations)

plot_buffer = pd.concat([data[['Date', 'Close']], future_forecast[['Date', 'Close']]], ignore_index=True)
plot_data = engineer_features(plot_buffer)

# -------------------------------------------------------------------------
# FEATURE IMPORTANCE VISUALIZATION (SIDEBAR)
# -------------------------------------------------------------------------
importances = final_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

fig_imp = go.Figure(go.Bar(
    x=importance_df['Importance'],
    y=importance_df['Feature'],
    orientation='h',
    marker_color='teal',
    opacity=0.8
))
fig_imp.update_layout(
    title="Model's Current Feature Weights",
    title_font_size=14,
    margin=dict(l=0, r=0, t=30, b=0),
    height=250,
    xaxis_title="Relative Importance Weight",
    yaxis_title=None,
    plot_bgcolor='white',
    paper_bgcolor='white'
)
fig_imp.update_xaxes(showgrid=True, gridcolor='rgba(230,230,230,0.5)')
st.sidebar.divider()
st.sidebar.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

# -----------------------------------------------------------------------------
# MASTER DASHBOARD VISUALIZATION
# -----------------------------------------------------------------------------
st.subheader("📈 AI Unified Technical & Forecast Dashboard")

with st.container():
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04, 
        subplot_titles=(
            f'{selected_stock} Institutional Forecast, Confidence & Monte Carlo', 
            'Bollinger Bands', 
            'RSI', 
            'MACD'
        ),
        row_heights=[0.5, 0.15, 0.15, 0.2] 
    )

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical Data', mode='lines', line=dict(color='black', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=future_forecast['Date'], y=future_forecast['Close'], name='AI Baseline Forecast', mode='lines', line=dict(color='blue', width=2.5)), row=1, col=1)

    for sim_key, path_prices in mc_paths_data.items():
        fig.add_trace(go.Scatter(
            x=future_forecast['Date'],
            y=path_prices,
            mode='lines',
            line=dict(color='rgba(0, 150, 255, 0.08)', width=1), 
            showlegend=False,
            hoverinfo='skip' 
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=future_forecast['Date'], 
        y=future_forecast['Upper_Bound'], 
        name='95% Confidence Upper', 
        line=dict(color='rgba(0,100,255,0.0)'),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=future_forecast['Date'], 
        y=future_forecast['Lower_Bound'], 
        name='95% Confidence Interval', 
        line=dict(color='rgba(0,100,255,0.0)'),
        fill='tonexty', 
        fillcolor='rgba(0,100,255,0.1)', 
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], name='Combined Price', line=dict(color='black', width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Upper_Band'], name='Upper BB', line=dict(color='rgba(0,0,255,0.3)', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Lower_Band'], name='Lower BB', line=dict(color='rgba(0,0,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,0,255,0.05)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['RSI'], name='RSI', line=dict(color='purple', width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,0,0,0.5)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,128,0,0.5)", row=3, col=1)

    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['MACD'], name='MACD', line=dict(color='blue', width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Signal_Line'], name='Signal', line=dict(color='red', width=1.5)), row=4, col=1)

    macd_hist = plot_data['MACD'] - plot_data['Signal_Line']
    macd_colors = ['rgba(0,128,0,0.6)' if val >= 0 else 'rgba(255,0,0,0.6)' for val in macd_hist]
    fig.add_trace(go.Bar(x=plot_data['Date'], y=macd_hist, name='Hist', marker_color=macd_colors), row=4, col=1)

    for r in range(1, 5):
        fig.add_vline(x=last_actual_date, line_dash="dot", line_color="green", opacity=0.6, row=r, col=1)
    fig.add_annotation(x=last_actual_date, y=1.05, yref="paper", text="Today", showarrow=False, font=dict(color="green", size=10), xanchor="left", row=1, col=1)

    fig.update_layout(
        height=900, 
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom",
            y=1.02, 
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        hovermode="x unified", 
        margin=dict(l=10, r=10, t=50, b=10), 
        plot_bgcolor='white', 
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.5)', zeroline=False)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
