import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots

# --- 1. ENHANCED DATA ENGINE ---
@st.cache_data
def get_technical_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['20STD'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['20STD'] * 2)

    # Fibonacci Retracement (Based on 1-Year High/Low)
    recent_period = df.tail(252)
    max_p = recent_period['High'].max()
    min_p = recent_period['Low'].min()
    diff = max_p - min_p
    df['Fib_0'] = max_p
    df['Fib_236'] = max_p - 0.236 * diff
    df['Fib_382'] = max_p - 0.382 * diff
    df['Fib_500'] = max_p - 0.500 * diff
    df['Fib_618'] = max_p - 0.618 * diff
    df['Fib_100'] = min_p

    return df.dropna()

# --- 2. INTERACTIVE DASHBOARD WITH TOOLTIPS ---
df = get_technical_data(ticker_input)
latest = df.iloc[-1]

st.subheader("📊 Key Technical Health Metrics")
m1, m2, m3, m4 = st.columns(4)

m1.metric("RSI (14D)", f"{latest['RSI']:.2f}", 
          help="**Relative Strength Index:** Measures speed/change of price. \n- **>70:** Overbought (Possible reversal) \n- **<30:** Oversold (Possible buy)")

m2.metric("MACD Hist", f"{latest['MACD_Hist']:.2f}",
          help="**MACD Histogram:** Shows the gap between the MACD and Signal line. \n- **Positive/Growing:** Strong bullish momentum. \n- **Negative/Shrinking:** Weakening trend.")

m3.metric("BB Bandwidth", f"{(latest['BB_Upper'] - latest['BB_Lower']):.2f}",
          help="**Bollinger Bandwidth:** Measures volatility. \n- **Narrowing:** 'The Squeeze' often precedes a major price breakout. \n- **Widening:** Increasing market volatility.")

m4.metric("Price vs Fib 61.8%", f"{latest['Close']:.2f}", delta=f"{(latest['Close'] - latest['Fib_618']):.2f}",
          help="**Fibonacci 61.8%:** The 'Golden Ratio' level. If the price holds above this during a dip, it indicates the long-term uptrend is still healthy.")

# --- 3. ADVANCED CHARTING ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, subplot_titles=('Price & BB/Fib', 'MACD', 'RSI'),
                    row_width=[0.2, 0.2, 0.6])

# Price & Bollinger Bands
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(dash='dash', color='gray')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(dash='dash', color='gray'), fill='tonexty'), row=1, col=1)

# Fibonacci Levels (Static lines for latest trend)
for lvl in ['Fib_0', 'Fib_236', 'Fib_382', 'Fib_500', 'Fib_618', 'Fib_100']:
    fig.add_trace(go.Scatter(x=[df['Date'].iloc[-30], df['Date'].iloc[-1]], 
                             y=[latest[lvl], latest[lvl]], 
                             mode='lines', name=lvl, line=dict(width=1, dash='dot')), row=1, col=1)

# MACD Histogram
colors = ['green' if x > 0 else 'red' for x in df['MACD_Hist']]
fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='MACD Hist', marker_color=colors), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

fig.update_layout(height=800, template="plotly_white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
