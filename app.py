import streamlit as st
from datetime import date
import yfinance as yf
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "1990-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Pro Dashboard", layout="wide")
st.title('🌳 Unified Technical Dashboard by S.Shah')

# -----------------------------------------------------------------------------
# SIDEBAR (CONFIGURATION)
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

test_years = st.sidebar.slider('Test Data Horizon (Years):', 1, 4, value=1)
n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)

test_days = test_years * 252
forecast_days = n_years * 252 

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
    """Calculates all technicals required for both the model and the plots."""
    df = df.copy()
    
    # Time Features
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    
    # Moving Averages & Bands
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Lags and Target for ML
    df['Lag_1_Close'] = df['Close'].shift(1)
    df['Lag_2_Close'] = df['Close'].shift(2)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Target_Return'] = df['Daily_Return'].shift(-1)
    
    return df

# Apply to all historical data
full_ml_data = engineer_features(data)

# Split into Train and Test sets
train_data = full_ml_data.iloc[:-test_days].dropna().copy()
test_data = full_ml_data.iloc[-test_days:].copy()

# -----------------------------------------------------------------------------
# XGBOOST MODEL TRAINING
# -----------------------------------------------------------------------------
features = ['Lag_1_Close', 'Lag_2_Close', 'SMA_10', 'SMA_20', 'MACD', 'RSI', 'DayOfYear', 'Month']
target = 'Target_Return'

model = xgb.XGBRegressor(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=5, 
    subsample=0.8,
    random_state=42
)

with st.spinner(f"Training XGBoost Model on {selected_stock}..."):
    model.fit(train_data[features], train_data[target])

# -----------------------------------------------------------------------------
# DYNAMIC ITERATIVE FORECASTING ENGINE
# -----------------------------------------------------------------------------
def generate_autoregressive_forecast(start_buffer, dates_to_predict):
    """Simulates future days one by one, recalculating technicals at each step."""
    buffer = start_buffer.copy()
    predictions = []
    
    for date_val in dates_to_predict:
        current_features_df = engineer_features(buffer)
        last_row = current_features_df.iloc[-1]
        
        X_pred = pd.DataFrame({
            'Lag_1_Close': [last_row['Lag_1_Close']],
            'Lag_2_Close': [last_row['Lag_2_Close']],
            'SMA_10': [last_row['SMA_10']],
            'SMA_20': [last_row['SMA_20']],
            'MACD': [last_row['MACD']],
            'RSI': [last_row['RSI']],
            'DayOfYear': [date_val.dayofyear],
            'Month': [date_val.month]
        })
        
        predicted_return = model.predict(X_pred)[0]
        new_close = last_row['Close'] * (1 + predicted_return)
        
        predictions.append({'Date': date_val, 'Close': new_close, 'Type': 'Forecast'})
        
        new_row = pd.DataFrame({'Date': [date_val], 'Close': [new_close]})
        buffer = pd.concat([buffer, new_row], ignore_index=True).tail(60) # Keep buffer light
        
    return pd.DataFrame(predictions)

with st.spinner("Simulating Test Data & Future Trajectory..."):
    # 1. Simulate the Test Period
    test_start_buffer = train_data[['Date', 'Close']].tail(60).copy()
    test_dates = test_data['Date'].tolist()
    test_forecast = generate_autoregressive_forecast(test_start_buffer, test_dates)
    
    # 2. Simulate the Future Period
    future_start_buffer = full_ml_data[['Date', 'Close']].tail(60).copy()
    last_actual_date = future_start_buffer['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    future_forecast = generate_autoregressive_forecast(future_start_buffer, future_dates)

# Combine for visualization
all_forecasts = pd.concat([test_forecast, future_forecast], ignore_index=True)

# Generate final plot data by running technicals over the historically connected forecast
combined_price_data = pd.concat([train_data[['Date', 'Close']], all_forecasts[['Date', 'Close']]], ignore_index=True)
plot_data = engineer_features(combined_price_data)

# -----------------------------------------------------------------------------
# CALCULATE ACCURACY METRICS ON TEST DATA
# -----------------------------------------------------------------------------
# Merge actuals with test forecast to calculate errors
eval_df = test_data[['Date', 'Close']].merge(test_forecast[['Date', 'Close']], on='Date', suffixes=('_Actual', '_Pred'))

mae = np.mean(np.abs(eval_df['Close_Actual'] - eval_df['Close_Pred']))
mape = np.mean(np.abs((eval_df['Close_Actual'] - eval_df['Close_Pred']) / eval_df['Close_Actual'])) * 100

st.subheader(f"Model Accuracy (Against Last {test_years} Years Held-Out Data)")
col1, col2, col3 = st.columns(3)
current_price = data['Close'].iloc[-1]

col1.metric("Current Known Price", f"${current_price:.2f}")
col2.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
col3.metric("MAPE (Percentage Error)", f"{mape:.2f}%")

st.divider()

# -----------------------------------------------------------------------------
# MASTER DASHBOARD (ALL IN ONE)
# -----------------------------------------------------------------------------
fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=(f'{selected_stock} XGBoost Forecast & Price', 'Bollinger Bands', 'RSI', 'MACD'),
    row_heights=[0.4, 0.2, 0.2, 0.2]
)

# --- ROW 1: PRICE & FORECAST ---
fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Close'], name='Train Data', mode='lines', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name='Test Data (Actual)', mode='lines', line=dict(color='rgba(0,0,0,0.3)')), row=1, col=1)
fig.add_trace(go.Scatter(x=test_forecast['Date'], y=test_forecast['Close'], name='Test Forecast Simulation', mode='lines', line=dict(color='orange', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=future_forecast['Date'], y=future_forecast['Close'], name='Future Forecast', mode='lines', line=dict(color='blue', width=2)), row=1, col=1)

split_date = train_data['Date'].iloc[-1]
fig.add_vline(x=split_date, line_dash="dash", line_color="gray", row=1, col=1)
fig.add_annotation(
    x=split_date, y=1.05, yref="paper", text="Train/Test Split",
    showarrow=False, font=dict(color="gray"), xanchor="left", row=1, col=1
)

# --- ROW 2: BOLLINGER BANDS ---
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], name='Combined Price', line=dict(color='black', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Upper_Band'], name='Upper BB', line=dict(color='rgba(0,0,255,0.3)', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Lower_Band'], name='Lower BB', line=dict(color='rgba(0,0,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,0,255,0.05)'), row=2, col=1)

# --- ROW 3: RSI ---
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# --- ROW 4: MACD ---
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Signal_Line'], name='Signal', line=dict(color='red')), row=4, col=1)
colors = ['green' if val >= 0 else 'red' for val in (plot_data['MACD'] - plot_data['Signal_Line']).dropna()]
# Align colors with the non-NaN MACD subset
macd_dates = plot_data['Date'][plot_data['MACD'].notna()]
fig.add_trace(go.Bar(x=macd_dates, y=(plot_data['MACD'] - plot_data['Signal_Line']).dropna(), name='Hist', marker_color=colors), row=4, col=1)

# Add today line to all subplots
fig.add_vline(x=last_actual_date, line_dash="dot", line_color="green", row=1, col=1)
fig.add_vline(x=last_actual_date, line_dash="dot", line_color="green", row=2, col=1)
fig.add_vline(x=last_actual_date, line_dash="dot", line_color="green", row=3, col=1)
fig.add_vline(x=last_actual_date, line_dash="dot", line_color="green", row=4, col=1)

fig.update_layout(height=1200, showlegend=True, title_text="Unified Technical & XGBoost Forecast Dashboard")
fig.update_xaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)
