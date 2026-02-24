# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Import your modules
from bayesian_optimizer import BayesianOptimizer
from kelly import KellySizer
from factor_model import MacroFactorModel
from rl_execution import RLExecutionAgent
from allocator import RiskParityAllocator
from heston import HestonMonteCarlo
from transformer_alpha import TransformerAlpha
from forecast_metrics import ForecastMetrics

# --- Sidebar: Data selection ---
st.sidebar.header("Portfolio Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with asset returns", type=["csv"])

if uploaded_file is not None:
    returns = pd.read_csv(uploaded_file, index_col=0)
    assets = list(returns.columns)
    st.subheader("Selected Asset Returns")
    st.dataframe(returns.head())
else:
    # Default random generation
    asset_list = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "BTC-USD", "ETH-USD"]
    selected_assets = st.sidebar.multiselect(
        "Select assets to include", asset_list, default=asset_list[:3]
    )
    if not selected_assets:
        st.warning("Select at least one asset!")
        st.stop()
    returns = pd.DataFrame(np.random.randn(100, len(selected_assets)), columns=selected_assets)
    assets = list(returns.columns)
    st.subheader("Generated Asset Returns")
    st.dataframe(returns.head())

# --- Sidebar: Parameters ---
st.sidebar.header("Simulation Parameters")
window = st.sidebar.number_input("Lookback window for metrics", min_value=10, max_value=250, value=60)

# --- Step 1: Forecasting ---
st.header("Forecasting & Alpha Signals")
transformer = TransformerAlpha()
forecasted_returns = transformer.predict(returns)  # assume method returns a DataFrame
st.write("Forecasted returns (first 5 rows):")
st.dataframe(forecasted_returns.head())

# --- Step 2: Risk Model ---
st.header("Risk Model")
factor_model = MacroFactorModel()
factor_cov = factor_model.fit(forecasted_returns)
st.write("Factor covariance matrix:")
st.dataframe(factor_cov)

# --- Step 3: Portfolio Allocation ---
st.header("Portfolio Allocation")
# Bayesian optimization of hyperparameters for allocation
optimizer = BayesianOptimizer()
opt_params = optimizer.optimize(forecasted_returns)

# Kelly optimal sizing
kelly = KellySizer()
kelly_weights = kelly.compute(forecasted_returns)

# Risk parity allocation
allocator = RiskParityAllocator()
risk_parity_weights = allocator.allocate(forecasted_returns)

st.subheader("Optimized Weights")
weights_df = pd.DataFrame({
    "Kelly": kelly_weights,
    "Risk Parity": risk_parity_weights
}, index=assets)
st.dataframe(weights_df)

# --- Step 4: Backtesting ---
st.header("Backtesting & Performance Metrics")
heston_model = HestonMonteCarlo()
simulated_prices = heston_model.simulate(returns)

metrics = ForecastMetrics()
performance = metrics.evaluate(simulated_prices, weights_df["Kelly"])
st.write("Performance Metrics for Kelly Portfolio:")
st.dataframe(performance)

# --- Step 5: Reinforcement Learning Execution ---
st.header("Execution Simulation")
rl_agent = RLExecutionAgent()
executed_trades = rl_agent.simulate_trades(weights_df["Kelly"], simulated_prices)
st.write("Executed Trades Sample:")
st.dataframe(executed_trades.head())
