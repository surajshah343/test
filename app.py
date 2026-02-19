import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_components_plotly
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Pro Stock Forecast App", layout="wide")
st.title('📈 Stock Dashboard by S. Shah')

# -----------------------------------------------------------------------------
# SIDEBAR (CONFIGURATION & TUNING)
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA")
selected_stock = ticker_input.upper()

n_years = st.sidebar.slider('Future Forecast Horizon (Years):', 1, 4, value=1)
period = n_years * 252 

test_years = st.sidebar.slider('Historical Test Period (Years):', 1, 10, value=6)

st.sidebar.divider()

st.sidebar.subheader("Prophet Model Tuning")
st.sidebar.markdown("""
Tweak these to lower the MAE/MAPE error scores.
""")
cps = st.sidebar.slider("Changepoint Prior Scale (Flexibility)", 
                        min_value=0.001, max_value=0.500, value=0.050, step=0.001,
                        help="Higher = more flexible trend. Lower = stiffer trend.")

sps = st.sidebar.slider("Seasonality Prior Scale", 
                        min_value=0.01, max_value=15.00, value=10.00, step=0.01,
                        help="Higher = model fits larger seasonal fluctuations. Lower = dampens seasonal effects.")

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
# FORECASTING LOGIC (HOLIDAYS, WEEKENDS OFF & DYNAMIC SPLIT)
# -----------------------------------------------------------------------------
df_prophet = data[['Date','Close']].rename(columns={"Date": "ds", "Close": "y"})

test_days = test_years * 252

if len(df_prophet) <= test_days:
    st.error(f"Not enough historical data to hold out {test_years} years. Please reduce the Test Period.")
    st.stop()

train_data = df_prophet.iloc[:-test_days]
test_data = df_prophet.iloc[-test_days:]

m = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
m.add_country_holidays(country_name='US')
m.fit(train_data)

total_periods = test_days + period
future = m.make_future_dataframe(periods=total_periods, freq='B')
forecast = m.predict(future)

# -----------------------------------------------------------------------------
# CALCULATE ACCURACY METRICS ON TEST DATA
# -----------------------------------------------------------------------------
eval_df = test_data.merge(forecast[['ds', 'yhat']], on='ds', how='inner')

mae = np.mean(np.abs(eval_df['y'] - eval_df['yhat']))
mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100

st.subheader(f"Model Accuracy (Against Last {test_years} Years Held-Out Data)")
col1, col2, col3 = st.columns(3)
current_price = data['Close'].iloc[-1]

col1.metric("Current Known Price", f"${current_price:.2f}")
col2.metric("Mean Absolute Error (MAE)", f"${mae:.2f}", help=f"Average dollar amount the prediction was off over the last {test_years} years.")
col3.metric("MAPE (Percentage Error)", f"{mape:.2f}%", help=f"Average percentage the prediction was off over the last {test_years} years.")

st.divider()

# -----------------------------------------------------------------------------
# EXTEND TECHNICAL INDICATORS INTO THE FUTURE
# -----------------------------------------------------------------------------
def calculate_technicals(df):
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 1. Grab historical facts
historical_df = data[['Date', 'Close']].copy()

# 2. Grab strictly future predictions (dates beyond today)
last_actual_date = historical_df['Date'].max()
future_df = forecast[forecast['ds'] > last_actual_date][['ds', 'yhat']].copy()
future_df = future_df.rename(columns={'ds': 'Date', 'yhat': 'Close'})

# 3. Stitch them together to create one continuous timeline
extended_data = pd.concat([historical_df, future_df], ignore_index=True)

# 4. Run the complex indicator math on the seamlessly stitched timeline
extended_data = calculate_technicals(extended_data)

# -----------------------------------------------------------------------------
# MASTER DASHBOARD (ALL IN ONE)
# -----------------------------------------------------------------------------
fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=(f'{selected_stock} Forecast & Price', 'Bollinger Bands (Extended)', 'RSI (Extended)', 'MACD (Extended)'),
    row_heights=[0.4, 0.2, 0.2, 0.2]
)

# --- ROW 1: PROPHET FORECAST ---
fig.add_trace(go.Scatter(x=train_data['ds'], y=train_data['y'], name='Train Data', mode='markers', marker=dict(color='black', size=4)), row=1, col=1)
fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['y'], name='Test Data (Actual)', mode='markers', marker=dict(color='orange', size=5)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet Prediction', mode='lines', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', name='Confidence'), row=1, col=1)

split_date = train_data['ds'].iloc[-1]
fig.add_vline(x=split_date, line_dash="dash", line_color="gray", row=1, col=1)
fig.add_annotation(
    x=split_date, y=1.05, yref="paper", text="Train/Test Split",
    showarrow=False, font=dict(color="gray"), xanchor="left", row=1, col=1
)

# --- ROW 2: BOLLINGER BANDS (Using extended_data) ---
fig.add_trace(go.Scatter(x=extended_data['Date'], y=extended_data['Close'], name='Combined Price', line=dict(color='black', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=extended_data['Date'], y=extended_data['Upper_Band'], name='Upper BB', line=dict(color='rgba(0,0,255,0.3)', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=extended_data['Date'], y=extended_data['Lower_Band'], name='Lower BB', line=dict(color='rgba(0,0,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,0,255,0.05)'), row=2, col=1)

# --- ROW 3: RSI (Using extended_data) ---
fig.add_trace(go.Scatter(x=extended_data['Date'], y=extended_data['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# --- ROW 4: MACD (Using extended_data) ---
fig.add_trace(go.Scatter(x=extended_data['Date'], y=extended_data['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
fig.add_trace(go.Scatter(x=extended_data['Date'], y=extended_data['Signal_Line'], name='Signal', line=dict(color='red')), row=4, col=1)
colors = ['green' if val >= 0 else 'red' for val in (extended_data['MACD'] - extended_data['Signal_Line'])]
fig.add_trace(go.Bar(x=extended_data['Date'], y=(extended_data['MACD'] - extended_data['Signal_Line']), name='Hist', marker_color=colors), row=4, col=1)

# Mark where the future begins in the bottom charts
fig.add_vline(x=last_actual_date, line_dash="dot", line_color="gray", row=2, col=1)
fig.add_vline(x=last_actual_date, line_dash="dot", line_color="gray", row=3, col=1)
fig.add_vline(x=last_actual_date, line_dash="dot", line_color="gray", row=4, col=1)

fig.update_layout(height=1200, showlegend=True, title_text="Unified Technical & Forecast Dashboard")
fig.update_xaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# FORECAST COMPONENTS (TREND, WEEKLY, YEARLY, HOLIDAYS)
# -----------------------------------------------------------------------------
st.subheader(f"🔍 {selected_stock} Forecast Components")
st.markdown("Breakdown of the overall trajectory, recurring seasonal patterns, and holiday impacts discovered by the model.")

fig_comp = plot_components_plotly(m, forecast)
st.plotly_chart(fig_comp, use_container_width=True)

# -----------------------------------------------------------------------------
# RAW DATA TAB
# -----------------------------------------------------------------------------
with st.expander("📝 View Raw Historical Data"):
    # We display the original historical data here, untainted by future predictions
    st.dataframe(data.tail(50))
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Historical CSV", data=csv, file_name=f"{selected_stock}_history.csv", mime="text/csv")
