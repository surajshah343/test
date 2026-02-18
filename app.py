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

st.set_page_config(page_title="Smart Stock Forecast", layout="wide")
st.title('📈 Smart Stock Forecast App')

# -----------------------------------------------------------------------------
# SIDEBAR: USER INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

# Dynamic Ticker Search
ticker_input = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** This tool uses Facebook Prophet for long-term trend forecasting and Technical Indicators for short-term momentum."
)

# -----------------------------------------------------------------------------
# DATA LOADING (With Caching & Error Handling)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        
        # FIX: Flatten MultiIndex columns if present (yfinance bug workaround)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

if data is None or data.empty:
    st.error(f"Error: Could not find data for ticker '{selected_stock}'. Please check the spelling.")
    st.stop()

# -----------------------------------------------------------------------------
# TABBED INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview & Forecast", "🛠️ Technical Analysis", "📝 Raw Data"])

# =============================================================================
# TAB 1: OVERVIEW & FORECAST (Your Original Logic)
# =============================================================================
with tab1:
    st.subheader(f'Forecast for {selected_stock}')

    # 1. Prophet Forecasting Logic (Strictly Unchanged)
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # 2. Plotting the Forecast
    st.write(f'**Price Prediction for next {n_years} years**')
    fig1 = plot_plotly(m, forecast)
    
    # Customizing the plot layout for better readability
    fig1.update_layout(
        xaxis_title="Date (Time)",
        yaxis_title="Stock Price (USD)",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Explanation for the user
    with st.expander("ℹ️ How to read this graph"):
        st.markdown("""
        * **X-Axis (Horizontal):** Represents Time (Years).
        * **Y-Axis (Vertical):** Represents the Stock Price in Dollars.
        * **Black Dots:** The actual historical price of the stock.
        * **Blue Line:** The model's predicted "Trend" line.
        * **Light Blue Shading:** The "Confidence Interval". The model is 80% sure the real price will fall inside this shaded area. Wider shading means more uncertainty.
        """)

    # 3. Forecast Components
    st.write("**Forecast Components**")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
    
    with st.expander("ℹ️ How to read components"):
        st.markdown("""
        * **Trend:** Shows the general long-term direction (up or down).
        * **Weekly:** Shows which days of the week are usually strongest (e.g., "Fridays are usually good").
        * **Yearly:** Shows seasonal patterns (e.g., "Stocks often rally in December").
        """)

# =============================================================================
# TAB 2: TECHNICAL ANALYSIS (New Feature)
# =============================================================================
with tab2:
    st.subheader("Technical Analysis Dashboard")
    st.caption("Detailed analysis of current market momentum.")

    # Calculate Indicators
    # 1. RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. Simple Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Get latest values
    latest_close = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    latest_sma_50 = data['SMA_50'].iloc[-1]
    latest_sma_200 = data['SMA_200'].iloc[-1]

    # DETERMINE SIGNAL
    signal = "NEUTRAL"
    color = "off"
    
    # Simple logic for signal
    if latest_rsi < 30:
        signal = "STRONG BUY (Oversold)"
    elif latest_rsi > 70:
        signal = "STRONG SELL (Overbought)"
    elif latest_close > latest_sma_200 and latest_close > latest_sma_50:
        signal = "BUY (Bullish Trend)"
    elif latest_close < latest_sma_200 and latest_close < latest_sma_50:
        signal = "SELL (Bearish Trend)"

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${latest_close:.2f}")
    col2.metric("RSI (14-day)", f"{latest_rsi:.2f}", help="<30 is Oversold, >70 is Overbought")
    col3.metric("Signal", signal)

    # Plot Technicals
    st.write("### Price vs Moving Averages")
    fig_tech = go.Figure()
    fig_tech.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', line=dict(color='blue')))
    fig_tech.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='50-Day SMA', line=dict(color='orange')))
    fig_tech.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], name='200-Day SMA', line=dict(color='red')))
    
    fig_tech.update_layout(title='Moving Average Crossover', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_tech, use_container_width=True)
    
    with st.expander("ℹ️ What is this chart telling me?"):
        st.markdown("""
        * **Golden Cross:** When the Orange Line (50-Day) crosses *above* the Red Line (200-Day), it's often a signal a Bull Market is starting.
        * **Death Cross:** When the Orange Line crosses *below* the Red Line, it often signals a Bear Market.
        """)

# =============================================================================
# TAB 3: RAW DATA
# =============================================================================
with tab3:
    st.subheader('Recent Data')
    st.dataframe(data.tail(10))
    
    # Download Button
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Data as CSV",
        data=csv,
        file_name=f"{selected_stock}_data.csv",
        mime="text/csv",
    )
