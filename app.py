import streamlit as st
from datetime import date
import yfinance as yf
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
import calendar

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "1990-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Pro Dashboard", layout="wide")
st.title('🌳 Auto-Optimized Technical Dashboard by S. Shah')

# -----------------------------------------------------------------------------
# SIDEBAR (CONFIGURATION)
# -----------------------------------------------------------------------------
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
    """
    Calculates technicals. Features must be mathematically stationary (percentages) 
    so XGBoost can extrapolate future trends without hitting tree boundaries.
    """
    df = df.copy()
    
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    
    # Standard Indicators (Used for plotting)
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
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ML Features (Stationary/Relative)
    df['Lag_1_Ret'] = df['Close'] / df['Close'].shift(1) - 1
    df['Lag_2_Ret'] = df['Close'] / df['Close'].shift(2) - 1
    df['SMA_10_Pct'] = df['SMA_10'] / df['Close'] - 1
    df['SMA_20_Pct'] = df['SMA_20'] / df['Close'] - 1
    df['MACD_Pct'] = df['MACD'] / df['Close']
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Target_Return'] = df['Daily_Return'].shift(-1)
    
    return df

# Create raw dataset, then drop NaNs ONLY for the ML training set
all_data_engineered = engineer_features(data)

# Features are now strictly relative distances, not absolute prices
features = ['Lag_1_Ret', 'Lag_2_Ret', 'SMA_10_Pct', 'SMA_20_Pct', 'MACD_Pct', 'RSI', 'DayOfYear', 'Month']
target = 'Target_Return'

# ML dataset safe for training (drops the final row where Target_Return is NaN)
full_ml_data = all_data_engineered.dropna(subset=features + [target]).copy()

# -----------------------------------------------------------------------------
# AUTOREGRESSIVE FORECAST FUNCTION
# -----------------------------------------------------------------------------
def generate_autoregressive_forecast(trained_model, start_buffer, dates_to_predict):
    """Simulates future days iteratively, maintaining a 300-day buffer for EMA stability."""
    buffer = start_buffer.copy()
    predictions = []
    
    for date_val in dates_to_predict:
        # Calculate features on the buffer. 
        current_features_df = engineer_features(buffer) 
        last_row = current_features_df.iloc[-1]
        
        X_pred = pd.DataFrame({
            'Lag_1_Ret': [last_row['Lag_1_Ret']],
            'Lag_2_Ret': [last_row['Lag_2_Ret']],
            'SMA_10_Pct': [last_row['SMA_10_Pct']],
            'SMA_20_Pct': [last_row['SMA_20_Pct']],
            'MACD_Pct': [last_row['MACD_Pct']],
            'RSI': [last_row['RSI']],
            'DayOfYear': [date_val.dayofyear],
            'Month': [date_val.month]
        })
        
        predicted_return = trained_model.predict(X_pred)[0]
        new_close = last_row['Close'] * (1 + predicted_return)
        
        predictions.append({'Date': date_val, 'Close': new_close})
        
        new_row = pd.DataFrame({'Date': [date_val], 'Close': [new_close]})
        # Keep 300 days to ensure EMA mathematically converges and doesn't distort
        buffer = pd.concat([buffer, new_row], ignore_index=True).tail(300) 
        
    return pd.DataFrame(predictions)

# -----------------------------------------------------------------------------
# ML OPTIMIZATION ENGINE
# -----------------------------------------------------------------------------
def optimize_xgboost_horizon():
    test_years_grid = [1, 2, 3, 4]
    best_mape = float('inf')
    best_ty = 1
    best_test_forecast = None
    
    total_days = len(full_ml_data)
    
    for ty in test_years_grid:
        test_days_iter = ty * 252
        if total_days <= test_days_iter + 252:
            continue
            
        train_iter = full_ml_data.iloc[:-test_days_iter].copy()
        test_iter = full_ml_data.iloc[-test_days_iter:].copy()
        
        model_iter = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model_iter.fit(train_iter[features], train_iter[target])
        
        # Pull raw price buffer exactly up to the split date
        split_date = train_iter['Date'].iloc[-1]
        start_buffer = data[data['Date'] <= split_date][['Date', 'Close']].tail(300).copy()
        test_dates = test_iter['Date'].tolist()
        
        forecast_iter = generate_autoregressive_forecast(model_iter, start_buffer, test_dates)
        
        eval_df = test_iter[['Date', 'Close']].merge(forecast_iter[['Date', 'Close']], on='Date', suffixes=('_Actual', '_Pred'))
        mape_iter = np.mean(np.abs((eval_df['Close_Actual'] - eval_df['Close_Pred']) / eval_df['Close_Actual'])) * 100
        
        if mape_iter < best_mape:
            best_mape = mape_iter
            best_ty = ty
            best_test_forecast = forecast_iter
            
    return best_ty, best_mape, best_test_forecast

with st.spinner("Running Grid Search to find optimal Test Horizon..."):
    optimal_test_years, final_mape, test_forecast = optimize_xgboost_horizon()

st.sidebar.divider()
st.sidebar.subheader("🤖 Auto-ML Tuning Active")
st.sidebar.info(f"""
**Optimal Test Horizon:** {optimal_test_years} Years  
**Minimized MAPE:** {final_mape:.2f}%
""")

# -----------------------------------------------------------------------------
# FINAL MODEL TRAINING & FUTURE FORECAST
# -----------------------------------------------------------------------------
test_days = optimal_test_years * 252
train_data = full_ml_data.iloc[:-test_days].copy()
test_data = full_ml_data.iloc[-test_days:].copy()

final_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)

with st.spinner("Training Final Model and Generating Future Forecast..."):
    final_model.fit(train_data[features], train_data[target])
    
    # Grab the true current day from raw data to seed the future forecast
    future_start_buffer = data[['Date', 'Close']].tail(300).copy()
    last_actual_date = future_start_buffer['Date'].iloc[-1]
    
    # Now perfectly aligns to tomorrow
    future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    
    future_forecast = generate_autoregressive_forecast(final_model, future_start_buffer, future_dates)

# Generate smooth plotting data 
all_forecasts = pd.concat([test_forecast, future_forecast], ignore_index=True)
combined_price_data = pd.concat([train_data[['Date', 'Close']], all_forecasts[['Date', 'Close']]], ignore_index=True)
plot_data = engineer_features(combined_price_data)

# -----------------------------------------------------------------------------
# SEASONALITY ANALYSIS & DASHBOARD 
# -----------------------------------------------------------------------------
# (Visuals remain the same)
st.subheader("🗓️ Historical Seasonality Analysis")
st.markdown("Average performance historically grouped by the day of the week and month of the year.")

seas_df = data.copy()
seas_df['Daily_Return'] = seas_df['Close'].pct_change() * 100
seas_df['DayOfWeek'] = seas_df['Date'].dt.day_name()
seas_df['Month'] = seas_df['Date'].dt.month_name()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
day_stats = seas_df.groupby('DayOfWeek')['Daily_Return'].mean().reindex(day_order).fillna(0)
day_colors = ['green' if val >= 0 else 'red' for val in day_stats]

month_order = list(calendar.month_name)[1:]
month_stats = seas_df.groupby('Month')['Daily_Return'].mean().reindex(month_order).fillna(0)
month_colors = ['green' if val >= 0 else 'red' for val in month_stats]

col_s1, col_s2 = st.columns(2)

with col_s1:
    fig_day = go.Figure(data=[go.Bar(x=day_stats.index, y=day_stats.values, marker_color=day_colors)])
    fig_day.update_layout(title="Average Return by Day of Week (%)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_day, use_container_width=True)

with col_s2:
    fig_month = go.Figure(data=[go.Bar(x=month_stats.index, y=month_stats.values, marker_color=month_colors)])
    fig_month.update_layout(title="Average Return by Month (%)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_month, use_container_width=True)

st.divider()

st.subheader("📈 Unified Technical & Forecast Dashboard")

fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=(f'{selected_stock} XGBoost Forecast & Price', 'Bollinger Bands', 'RSI', 'MACD'),
    row_heights=[0.4, 0.2, 0.2, 0.2]
)

# ROW 1: PRICE & FORECAST
fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Close'], name='Train Data', mode='lines', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name='Test Data (Actual)', mode='lines', line=dict(color='rgba(0,0,0,0.3)')), row=1, col=1)
fig.add_trace(go.Scatter(x=test_forecast['Date'], y=test_forecast['Close'], name='Test Forecast (Simulated)', mode='lines', line=dict(color='orange', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=future_forecast['Date'], y=future_forecast['Close'], name='Future Forecast', mode='lines', line=dict(color='blue', width=2)), row=1, col=1)

split_date = train_data['Date'].iloc[-1]
fig.add_vline(x=split_date, line_dash="dash", line_color="gray", row=1, col=1)
fig.add_annotation(x=split_date, y=1.05, yref="paper", text="Train/Test Split", showarrow=False, font=dict(color="gray"), xanchor="left", row=1, col=1)

# ROW 2: BOLLINGER BANDS
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], name='Combined Price', line=dict(color='black', width=1), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Upper_Band'], name='Upper BB', line=dict(color='rgba(0,0,255,0.3)', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Lower_Band'], name='Lower BB', line=dict(color='rgba(0,0,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,0,255,0.05)'), row=2, col=1)

# ROW 3: RSI
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# ROW 4: MACD
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Signal_Line'], name='Signal', line=dict(color='red')), row=4, col=1)

macd_hist = plot_data['MACD'] - plot_data['Signal_Line']
macd_colors = ['green' if val >= 0 else 'red' for val in macd_hist]
fig.add_trace(go.Bar(x=plot_data['Date'], y=macd_hist, name='Hist', marker_color=macd_colors), row=4, col=1)

# TODAY LINES
for r in range(1, 5):
    fig.add_vline(x=last_actual_date, line_dash="dot", line_color="green", row=r, col=1)
fig.add_annotation(x=last_actual_date, y=1.05, yref="paper", text="Today", showarrow=False, font=dict(color="green"), xanchor="left", row=1, col=1)

fig.update_layout(height=1200, showlegend=True)
fig.update_xaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)
