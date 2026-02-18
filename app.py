import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Pro Stock Forecast App", layout="wide")
st.title('📈 Stock Dashboard by S. Shah')

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4)
period = n_years * 365

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

data = load_data(selected_stock)

if data is None or data.empty:
    st.error(f"Error: Could not find data for '{selected_stock}'. Check ticker symbol.")
    st.stop()

# -----------------------------------------------------------------------------
# TECHNICAL INDICATORS
# -----------------------------------------------------------------------------
def calculate_technicals(df):
    # 1. MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 2. Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    
    # 3. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

data = calculate_technicals(data)

# -----------------------------------------------------------------------------
# FORECASTING LOGIC
# -----------------------------------------------------------------------------
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# -----------------------------------------------------------------------------
# DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
st.subheader(f"Unified Dashboard: {selected_stock}")

# --- 1. THE BIG INTERACTIVE CHART (Forecast + Technicals) ---
# Create 4 stacked subplots with shared X-axis
fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    subplot_titles=('Price Forecast', 'Bollinger Bands', 'RSI', 'MACD'),
    row_heights=[0.5, 0.2, 0.15, 0.15]
)

# ROW 1: Forecast
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name='Actual', mode='markers', marker=dict(color='black', size=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction', mode='lines', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', name='Confidence'), row=1, col=1)

# ROW 2: Bollinger Bands
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Price', line=dict(color='black', width=1), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Band'], name='Upper BB', line=dict(color='rgba(0,0,255,0.3)', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Band'], name='Lower BB', line=dict(color='rgba(0,0,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,0,255,0.05)'), row=2, col=1)

# ROW 3: RSI
fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# ROW 4: MACD
fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], name='Signal', line=dict(color='red')), row=4, col=1)
colors = ['green' if val >= 0 else 'red' for val in (data['MACD'] - data['Signal_Line'])]
fig.add_trace(go.Bar(x=data['Date'], y=(data['MACD'] - data['Signal_Line']), name='Hist', marker_color=colors), row=4, col=1)

fig.update_layout(height=1000, title_text=f"Technical & Forecast Analysis for {selected_stock}")
fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- 2. FORECAST COMPONENTS (The "Tables" you missed) ---
st.write("---")
st.header("Forecast Components (Trends & Seasonality)")
st.write("Below are the specific trends extracted by the model (Weekly, Yearly, and Overall Trend).")

# We render the matplotlib figure here
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

# --- 3. RAW DATA ---
with st.expander("View Raw Data"):
    st.dataframe(data.tail())
