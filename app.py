import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.title('Stock Forecast App (Statistically Sound Version)')

st.sidebar.header("Configuration")
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'NVDA', 'TSLA')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# -----------------------------------------------------------------------------
# 1. STATISTICAL VALIDATION (THE REALITY CHECK)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("1. Statistical Validation (Backtesting)")
st.markdown("""
To prevent **Data Leakage**, we split the data into a **Training Set** (Past) and a **Test Set** (Recent History).
The model is trained *only* on the past and must predict the recent history without seeing it.
""")

# User selects how many days to hide from the model for testing
test_days = st.slider('Days to hide for validation (Test Set size):', 30, 730, 365)

# Create the split
cutoff_date = data['Date'].iloc[-test_days]
train_df = data[data['Date'] < cutoff_date]
test_df = data[data['Date'] >= cutoff_date]

st.write(f"**Training cutoff:** {cutoff_date.date()} (Model does not see data after this)")
st.write(f"**Training samples:** {len(train_df)} | **Test samples:** {len(test_df)}")

# Prepare data for Prophet
df_train_prophet = train_df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_test_prophet = test_df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Fit the model on TRAINING data only
m_val = Prophet(daily_seasonality=False)
m_val.fit(df_train_prophet)

# Predict the TEST period
future_val = m_val.make_future_dataframe(periods=test_days)
forecast_val = m_val.predict(future_val)

# Filter forecast to only the test period for comparison
forecast_test_only = forecast_val[forecast_val['ds'] >= cutoff_date]

# --- PLOT VALIDATION RESULTS ---
fig_val = go.Figure()

# 1. Actual Test Data
fig_val.add_trace(go.Scatter(
    x=df_test_prophet['ds'], 
    y=df_test_prophet['y'], 
    mode='markers',
    name='Actual Prices (Truth)',
    marker=dict(color='blue', size=4)
))

# 2. Model Predictions
fig_val.add_trace(go.Scatter(
    x=forecast_test_only['ds'], 
    y=forecast_test_only['yhat'], 
    mode='lines', 
    name='Model Prediction',
    line=dict(color='red')
))

# 3. Confidence Interval
fig_val.add_trace(go.Scatter(
    x=forecast_test_only['ds'], 
    y=forecast_test_only['yhat_upper'],
    mode='lines', line=dict(width=0), showlegend=False
))
fig_val.add_trace(go.Scatter(
    x=forecast_test_only['ds'], 
    y=forecast_test_only['yhat_lower'],
    mode='lines', line=dict(width=0), 
    fill='tonexty', 
    fillcolor='rgba(255, 0, 0, 0.2)',
    name='Uncertainty Interval'
))

fig_val.update_layout(title="Backtesting: Predicted vs Actual", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_val, use_container_width=True)

# --- CALCULATE METRICS ---
mae = mean_absolute_error(df_test_prophet['y'], forecast_test_only['yhat'])
rmse = math.sqrt(mean_squared_error(df_test_prophet['y'], forecast_test_only['yhat']))

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")

st.info(f"""
**Interpretation:** On average, the model's prediction was off by **${mae:.2f}** during the test period. 
If this error is high relative to the stock price, the model (Prophet) may not be suitable for this asset.
""")

# -----------------------------------------------------------------------------
# 2. FUTURE FORECAST (FULL MODEL)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header(f"2. Future Forecast ({n_years} Years)")
st.markdown("Now we retrain the model on **ALL** available data to predict into the unknown future.")

# Prepare full dataset
df_full = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(daily_seasonality=False)
m.fit(df_full)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show forecast
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write(f'Forecast Plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Forecast Components")
# Note: plot_components returns a matplotlib figure
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
