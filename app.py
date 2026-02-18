import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Pro Stock Forecast App", layout="wide")
st.title('📈 Institutional-Grade Stock Dashboard')

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
# CALCULATE PROFESSIONAL INDICATORS (Technical Analysis)
# -----------------------------------------------------------------------------
def calculate_technicals(df):
    # 1. MACD (Trend Speedometer)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 2. Bollinger Bands (Volatility Rubber Band)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)

    # 3. VWAP (Institutional True Price)
    # Note: True VWAP is intraday. We approximate "Rolling VWAP" for daily charts.
    # We use (High + Low + Close) / 3 as the typical price
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    # We use a rolling window here to approximate the trend, or cumulative for the whole series
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    return df

data = calculate_technicals(data)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Prophet Forecast", "🧠 Pro Dashboard (Technical Analysis)", "📝 Raw Data"])

# =============================================================================
# TAB 1: ORIGINAL PROPHET FORECAST (LOGIC UNCHANGED)
# =============================================================================
with tab1:
    st.subheader(f'AI Forecast for {selected_stock}')
    
    # Prophet Logic
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Plot
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(
        title=f"Prophet Forecast: {n_years} Year Horizon",
        yaxis_title="Stock Price (USD)",
        xaxis_title="Date"
    )
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("See Forecast Components"):
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

# =============================================================================
# TAB 2: PRO DASHBOARD (NEW COMPLEX INDICATORS)
# =============================================================================
with tab2:
    st.subheader("Institutional Technical Analysis")
    st.write("These indicators help answer: *Is the price fair? Is it overextended? Is momentum shifting?*")

    # --- CHART 1: Price vs VWAP & Bollinger Bands ---
    st.write("#### 1. Volatility & Fair Value (Bollinger Bands + VWAP)")
    
    fig_bol = go.Figure()
    
    # Candlestick for Price
    fig_bol.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name='Price'
    ))
    
    # VWAP Line
    fig_bol.add_trace(go.Scatter(
        x=data['Date'], y=data['VWAP'],
        line=dict(color='orange', width=2),
        name='VWAP (Fair Value)'
    ))
    
    # Bollinger Bands
    fig_bol.add_trace(go.Scatter(
        x=data['Date'], y=data['Upper_Band'],
        line=dict(color='rgba(0,0,255,0.3)', width=1),
        name='Upper Band'
    ))
    fig_bol.add_trace(go.Scatter(
        x=data['Date'], y=data['Lower_Band'],
        line=dict(color='rgba(0,0,255,0.3)', width=1),
        fill='tonexty', # Fills area between Upper and Lower
        fillcolor='rgba(0,0,255,0.05)',
        name='Lower Band'
    ))

    fig_bol.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_bol, use_container_width=True)
    
    st.info("""
    **How to Read:**
    * **The Orange Line (VWAP):** If Price is ABOVE this, bulls are in control (but it might be expensive). If BELOW, bears are in control (or it's "cheap").
    * **The Blue Cloud (Bollinger Bands):** If the cloud is wide, the market is volatile. If the price touches the top edge, it's statistically "overextended" and might pull back.
    """)

    # --- CHART 2: MACD MOMENTUM ---
    st.write("#### 2. Momentum Speedometer (MACD)")
    
    fig_macd = go.Figure()
    
    # MACD Line
    fig_macd.add_trace(go.Scatter(
        x=data['Date'], y=data['MACD'],
        line=dict(color='blue', width=2),
        name='MACD (Fast)'
    ))
    
    # Signal Line
    fig_macd.add_trace(go.Scatter(
        x=data['Date'], y=data['Signal_Line'],
        line=dict(color='red', width=2),
        name='Signal (Slow)'
    ))
    
    # Histogram (Difference)
    colors = ['green' if val >= 0 else 'red' for val in (data['MACD'] - data['Signal_Line'])]
    fig_macd.add_trace(go.Bar(
        x=data['Date'], y=(data['MACD'] - data['Signal_Line']),
        marker_color=colors,
        name='Momentum Histogram'
    ))

    fig_macd.update_layout(height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_macd, use_container_width=True)

    st.info("""
    **How to Read:**
    * **Crossover:** When the Blue Line crosses ABOVE the Red Line, it's a **"BUY"** signal (momentum is shifting up).
    * **Histogram:** Green bars mean the uptrend is accelerating. Red bars mean the downtrend is accelerating.
    """)

# =============================================================================
# TAB 3: RAW DATA
# =============================================================================
with tab3:
    st.dataframe(data.tail(50))
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name=f"{selected_stock}_pro_data.csv", mime="text/csv")
