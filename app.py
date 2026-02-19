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
START = "2015-01-01" # Keep timeframe reasonable for ML relevance
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="XGBoost Dynamic Forecast", layout="wide")
st.title('🌳 XGBoost Dynamic Technical Forecast')
st.markdown("This model dynamically recalculates RSI, MACD, and SMAs at every future step to prevent flatlining.")

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
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
        return data[['Date', 'Close', 'Volume']].copy()
    except Exception:
        return None

data = load_data(selected_stock)

if data is None or data.empty:
    st.error(f"Error: Could not pull data for '{selected_stock}'.")
    st.stop()

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING FUNCTION
# -----------------------------------------------------------------------------
def engineer_features(df):
    """Calculates technicals and time features. Designed to run on historical AND rolling future data."""
    df = df.copy()
    
    # Time Features to prevent the model from getting lost
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    
    # Technicals
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Lags and Target
    df['Lag_1_Close'] = df['Close'].shift(1)
    df['Lag_2_Close'] = df['Close'].shift(2)
    
    # We predict the daily percentage return rather than raw dollar difference
    # This scales better for rapidly growing stocks like NVDA
    df['Daily_Return'] = df['Close'].pct_change()
    df['Target_Return'] = df['Daily_Return'].shift(-1)
    
    return df

# Apply to historical
ml_data = engineer_features(data).dropna()

# -----------------------------------------------------------------------------
# XGBOOST MODEL TRAINING
# -----------------------------------------------------------------------------
features = ['Lag_1_Close', 'Lag_2_Close', 'SMA_10', 'SMA_20', 'MACD', 'RSI', 'DayOfYear', 'Month']
target = 'Target_Return'

X = ml_data[features]
y = ml_data[target]

# We use the entire dataset to train so it has the most recent context
model = xgb.XGBRegressor(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=5, 
    subsample=0.8,
    random_state=42
)

with st.spinner("Training Dynamic XGBoost Model..."):
    model.fit(X, y)

# -----------------------------------------------------------------------------
# DYNAMIC ITERATIVE FORECASTING
# -----------------------------------------------------------------------------
st.subheader("Predicting Future Trajectory")

# Create a rolling buffer of the last 50 days to calculate technicals for the future
rolling_buffer = data[['Date', 'Close']].tail(50).copy()
future_dates = pd.date_range(start=rolling_buffer['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

future_predictions = []

progress_bar = st.progress(0)
status_text = st.empty()

for i, date_val in enumerate(future_dates):
    # 1. Recalculate all features on the current rolling buffer
    current_features_df = engineer_features(rolling_buffer)
    
    # 2. Extract the very last row of features to make today's prediction
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
    
    # 3. Predict the percentage return for the next day
    predicted_return = model.predict(X_pred)[0]
    
    # 4. Calculate the new closing price based on the predicted return
    current_close = last_row['Close']
    new_close = current_close * (1 + predicted_return)
    
    future_predictions.append({'Date': date_val, 'Predicted_Close': new_close})
    
    # 5. Append the new prediction to the rolling buffer for the next loop iteration
    new_row = pd.DataFrame({'Date': [date_val], 'Close': [new_close]})
    rolling_buffer = pd.concat([rolling_buffer, new_row], ignore_index=True)
    
    # Keep buffer manageable
    rolling_buffer = rolling_buffer.tail(50)
    
    # Update progress
    if i % 20 == 0:
        progress_bar.progress((i + 1) / forecast_days)
        status_text.text(f"Simulating day {i+1} of {forecast_days}...")

progress_bar.empty()
status_text.empty()

future_df = pd.DataFrame(future_predictions)

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
fig = go.Figure()

# Plot historical
fig.add_trace(go.Scatter(
    x=data['Date'], y=data['Close'], 
    mode='lines', name='Historical Close', line=dict(color='black', width=1.5)
))

# Plot XGBoost future
fig.add_trace(go.Scatter(
    x=future_df['Date'], y=future_df['Predicted_Close'], 
    mode='lines', name='Dynamic XGBoost Forecast', line=dict(color='orange', width=2)
))

# Add a visual separator
split_date = future_df['Date'].iloc[0]
fig.add_vline(x=split_date, line_dash="dash", line_color="gray")
fig.add_annotation(
    x=split_date, y=1.05, yref="paper", text="Today",
    showarrow=False, font=dict(color="gray"), xanchor="left"
)

fig.update_layout(
    title=f"XGBoost Dynamic Forecast for {selected_stock}", 
    height=600,
    xaxis_title="Date",
    yaxis_title="Stock Price ($)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
