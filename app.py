import streamlit as st
import os
import json
from datetime import date, datetime, timedelta
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="AI Quant Pro v6.0", layout="wide")
st.title('🧠 Financial AI: Quantitative Stress Test Framework')

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=3)
forecast_days = int(n_years * 252) 
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)

st.sidebar.markdown("---")
st.sidebar.subheader("📉 Scenario Analysis")
# STRESS TEST SLIDER
shock_pct = st.sidebar.slider("Day 1 Stress Test Shock (%)", min_value=-50, max_value=50, value=0, step=1, help="Force a sudden percentage move on Day 1 of the forecast to see how the AI handles the recovery.")

st.sidebar.markdown("---")
retrain_button = st.sidebar.button("🔄 Force Model Retrain")

# Version bump to v6 for clean state
MODEL_FILE = f"saved_models/{ticker_input}_v6.json"

# --- 1. DATA LOADING & TECHNICALS ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, start=START, end=TODAY)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA20'] = df['Close'].rolling(20).mean()
    df['stddev'] = df['Close'].rolling(20).std()
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    df['Upper'] = df['MA20'] + (df['stddev'] * 2)
    df['Lower'] = df['MA20'] - (df['stddev'] * 2)
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    return df.dropna().copy()

data = load_data(ticker_input)
if data is None: st.stop()

# --- 2. FEATURE ENGINEERING ---
def engineer_features(df):
    df = df.copy()
    df['Lag_1_Ret'] = df['Log_Ret'].shift(1)
    df['SMA_20_Pct'] = (df['MA20'] / df['Close']) - 1
    df['Target_Residual'] = df['Log_Ret'].shift(-1) - df['Log_Ret'].rolling(50).mean()
    df['DayOfYear'] = df['Date'].dt.dayofyear / 366.0
    return df.dropna().copy()

ml_data = engineer_features(data)
features = ['Lag_1_Ret', 'SMA_20_Pct', 'Vol_20', 'DayOfYear']
target = 'Target_Residual'

split_idx = int(len(ml_data) * 0.8)
train_set = ml_data.iloc[:split_idx]
test_set = ml_data.iloc[split_idx:]

# --- 3. TRAINING ---
if not os.path.exists(MODEL_FILE) or retrain_button:
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
    model.fit(train_set[features], train_set[target])
    model.save_model(MODEL_FILE)
    final_model = model
else:
    final_model = xgb.XGBRegressor()
    final_model.load_model(MODEL_FILE)

test_preds = final_model.predict(test_set[features])
hit_ratio = np.mean(np.sign(test_preds) == np.sign(test_set[target].values)) * 100
importance = dict(zip(features, final_model.feature_importances_))

# --- 4. STRESS-TEST ENABLED SIMULATION ---
@st.cache_data(show_spinner="Simulating Stochastic Paths...")
def run_simulation(_model, _historical_df, n_days, n_sims, ticker, shock):
    last_price = _historical_df['Close'].iloc[-1]
    last_date = _historical_df['Date'].iloc[-1]
    
    long_term_mu = _historical_df['Log_Ret'].tail(252).mean()
    sigma = _historical_df['Log_Ret'].tail(252).std()
    
    all_paths = np.zeros((n_days, n_sims))
    current_prices = np.full(n_sims, last_price)
    
    simulated_history = list(_historical_df['Close'].tail(20).values)
    current_log_ret = _historical_df['Log_Ret'].iloc[-1]
    
    for d in range(n_days):
        current_ma20 = np.mean(simulated_history[-20:])
        current_mean_price = np.mean(current_prices)
        next_date = last_date + pd.Timedelta(days=d+1)
        
        pred_feat = pd.DataFrame({
            'Lag_1_Ret': [current_log_ret],
            'SMA_20_Pct': [(current_ma20 / current_mean_price) - 1], 
            'Vol_20': [sigma],
            'DayOfYear': [next_date.dayofyear / 366.0]
        })
        alpha = _model.predict(pred_feat)[0]
        
        # INJECT DAY 1 STRESS TEST SHOCK
        if d == 0 and shock != 0:
            log_returns = np.full(n_sims, np.log(1 + (shock / 100.0)))
        else:
            shocks = np.random.normal(0, sigma, n_sims)
            log_returns = alpha + long_term_mu + shocks
            
        current_prices *= np.exp(log_returns)
        all_paths[d, :] = current_prices
        
        current_log_ret = np.mean(log_returns)
        simulated_history.append(current_mean_price)
        
    return all_paths

sim_results = run_simulation(final_model, ml_data, forecast_days, n_simulations, ticker_input, shock_pct)

final_prices = sim_results[-1, :]
initial_price = data['Close'].iloc[-1]
var_95_price = np.percentile(final_prices, 5)
var_95_pct = ((var_95_price - initial_price) / initial_price) * 100

# --- 5. TOP LEVEL UI METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${initial_price:.2f}")
m2.metric("Backtest Hit Ratio", f"{hit_ratio:.1f}%", help="Directional Accuracy on the Test Set.")
m3.metric("95% Horizon VaR", f"{var_95_pct:.1f}%", help=f"5% probability the asset drops below ${var_95_price:.2f}.")
m4.metric("Test MAE", f"{mean_absolute_error(test_set[target], test_preds):.5f}")

st.markdown("---")

# --- 6. TABBED LAYOUT FOR MAXIMUM VISIBILITY ---
tab1, tab2, tab3 = st.tabs(["🔮 Forecast & Stress Test", "🛠 Technical Regimes", "🧠 AI Feature Logic"])

with tab1:
    st.subheader(f"📈 Stochastic Projection ({n_years}Y)")
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=train_set['Date'], y=train_set['Close'], name='Training Data (Seen)', line=dict(color='#2980b9')))
    fig_main.add_trace(go.Scatter(x=test_set['Date'], y=test_set['Close'], name='Testing Data (Unseen)', line=dict(color='#e67e22')))

    future_dates = pd.date_range(ml_data['Date'].max(), periods=forecast_days + 1, freq='B')[1:]
    fig_main.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 97.5, axis=1), line=dict(width=0), showlegend=False))
    fig_main.add_trace(go.Scatter(x=future_dates, y=np.percentile(sim_results, 5, axis=1), line=dict(width=0), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.15)', name='95% VaR Boundary'))
    fig_main.add_trace(go.Scatter(x=future_dates, y=np.median(sim_results, axis=1), name='AI Median Forecast', line=dict(color='#2ecc71', width=3)))

    fig_main.add_shape(type="line", x0=train_set['Date'].iloc[-1], x1=train_set['Date'].iloc[-1], y0=0, y1=1, yref="paper", line=dict(color="Red", width=1, dash="dash"))
    fig_main.add_annotation(x=train_set['Date'].iloc[-1], y=1.05, yref="paper", text="Train/Test Split", showarrow=False, font=dict(color="red"))

    fig_main.update_layout(template="plotly_white", hovermode="x unified", xaxis_rangeslider_visible=True, height=650)
    st.plotly_chart(fig_main, use_container_width=True)

with tab2:
    st.subheader("🛠 Technical Regime Analysis (Last 500 Days)")
    tech_view = ml_data.tail(500)
    fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])

    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Upper'], line=dict(color='rgba(173, 216, 230, 0.5)'), name='Upper Band'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Lower'], line=dict(color='rgba(173, 216, 230, 0.5)'), fill='tonexty', name='Lower Band'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Close'], line=dict(color='#2c3e50'), name='Close'), row=1, col=1)

    macd_colors = ['#26a69a' if x > 0 else '#ef5350' for x in tech_view['MACD_Hist']]
    fig_tech.add_trace(go.Bar(x=tech_view['Date'], y=tech_view['MACD_Hist'], name='MACD Hist', marker_color=macd_colors), row=2, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['MACD'], name='MACD', line=dict(color='#2980b9')), row=2, col=1)

    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['RSI'], name='RSI', line=dict(color='#8e44ad')), row=3, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=3, col=1)

    fig_tech.update_layout(height=700, showlegend=False, template="plotly_white")
    st.plotly_chart(fig_tech, use_container_width=True)

with tab3:
    st.subheader("🧠 Feature Intelligence & AI Logic")
    top_feature = max(importance, key=importance.get)
    current_sentiment = "Bullish" if np.mean(test_preds[-10:]) > 0 else "Bearish"

    col_chart, col_text = st.columns([1, 1])

    with col_chart:
        importance_df = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())}).sort_values(by='Importance')
        fig_imp = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h', marker_color='#34495e'))
        fig_imp.update_layout(title="Relative Feature Importance (Gain)", template="plotly_white", height=350, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_text:
        st.write("### AI Regime Summary")
        st.write(f"**Current Directional Bias:** {current_sentiment}")
        
        analysis = f"The XGBoost model for **{ticker_input}** is currently heavily influenced by **{top_feature}**. "
        if top_feature == 'Vol_20':
            analysis += "This indicates a **Volatility Regime** where price swings and risk metrics are the primary predictors of future drift. "
        elif top_feature == 'SMA_20_Pct':
            analysis += "The AI is relying on **Mean Reversion**, heavily weighting how far the price deviates from its 20-day moving average. "
        elif top_feature == 'DayOfYear':
            analysis += "Historical **Seasonality** is dominating the model, implying the stock is rigidly following its typical annual patterns. "
        elif top_feature == 'Lag_1_Ret':
            analysis += "The model is operating in a **Momentum Regime**, placing the highest value on immediate, day-to-day price inertia. "
            
        analysis += f"\n\nWith an out-of-sample directional hit ratio of **{hit_ratio:.1f}%**, the model shows "
        analysis += "moderate" if hit_ratio < 53 else "strong"
        analysis += " confidence in this predictive structure."
        
        st.info(analysis)
