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
START = "2010-01-01" # Trimmed data for ML performance
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="XGBoost Stock Forecast", layout="wide")
st.title('🌳 XGBoost Technical Forecast Dashboard')

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
forecast_days = n_years * 252 

# -----------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

if data is None or data.empty:
    st.error("Error loading data.")
    st.stop()

def calculate_technicals(df):
    df = df.copy()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Target Variable: Daily price difference
    df['Price_Diff'] = df['Close'].diff()
    
    # Lagged features (yesterday's data predicting today's diff)
    df['Lag_Close'] = df['Close'].shift(1)
    df['Lag_RSI'] = df['RSI'].shift(1)
    df['Lag_MACD'] = df['MACD'].shift(1)
    df['Target_Diff'] = df['Price_Diff'].shift(-1) # What we are trying to predict
    
    return df.dropna()

ml_data = calculate_technicals(data)

# -----------------------------------------------------------------------------
# XGBOOST MODEL TRAINING
# -----------------------------------------------------------------------------
features = ['Lag_Close', 'Lag_RSI', 'Lag_MACD']
target = 'Target_Diff'

# Split Data (Leave out last year for testing)
test_size = 252
train = ml_data.iloc[:-test_size]
test = ml_data.iloc[-test_size:]

X_train = train[features]
y_train = train[target]

model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=4, 
    random_state=42
)

with st.spinner("Training XGBoost Model..."):
    model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# ITERATIVE FORECASTING
# -----------------------------------------------------------------------------
st.subheader("Predicting Future Prices")

# We start from the very last known row of our data
current_data = ml_data.iloc[-1:].copy()
future_dates = pd.date_range(start=current_data['Date'].iloc[0] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

future_predictions = []
current_close = current_data['Close'].iloc[0]
current_rsi = current_data['RSI'].iloc[0]
current_macd = current_data['MACD'].iloc[0]

# Generate future steps one day at a time
for date in future_dates:
    # Build feature row for prediction
    X_pred = pd.DataFrame({'Lag_Close': [current_close], 'Lag_RSI': [current_rsi], 'Lag_MACD': [current_macd]})
    
    # Predict the daily difference
    predicted_diff = model.predict(X_pred)[0]
    
    # Calculate new close
    new_close = current_close + predicted_diff
    future_predictions.append({'Date': date, 'Predicted_Close': new_close})
    
    # Update variables for next loop iteration
    # (In a true robust model, you'd recalculate MACD/RSI accurately over a rolling window, 
    # but we hold them slightly static/decayed here for performance in Streamlit)
    current_close = new_close

future_df = pd.DataFrame(future_predictions)

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
fig = go.Figure()

# Plot historical
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Close', line=dict(color='black')))

# Plot XGBoost future
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'], mode='lines', name='XGBoost Forecast', line=dict(color='orange', width=2)))

fig.update_layout(title="XGBoost Iterative Forecast (Predicting Daily Returns)", height=600)
st.plotly_chart(fig, use_container_width=True)
