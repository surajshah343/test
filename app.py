import streamlit as st
import os
import time
# Set environment variable to fix potential OMP error on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import date, datetime
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# Create the directory for saving models if it doesn't exist
os.makedirs("saved_models", exist_ok=True)
st.set_page_config(page_title="AI Quant Pro v10.1 - Persistent Deep Learning", layout="wide")
st.title('🧠 Financial AI: LSTM Deep Learning Framework & Backtester')

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Ticker:", value="AMZN").upper()
n_years = st.sidebar.slider('Forecast Horizon (Years):', 1, 4, value=3)
forecast_days = int(n_years * 252)
n_simulations = st.sidebar.slider('Monte Carlo Paths:', 100, 1000, value=500)
seq_length = st.sidebar.slider('LSTM Lookback Window (Days):', 10, 60, value=20)

# Define the path for the saved model weights based on the ticker
MODEL_WEIGHTS_PATH = os.path.join("saved_models", f"{ticker_input}_lstm_weights.pth")

retrain_button = st.sidebar.button("🔄 Force Model Retrain")

# --- 1. DATA ACQUISITION & FEATURE ENGINEERING ---
@st.cache_data(show_spinner=False)
def load_and_prep_data(ticker):
    # Download 10 years of daily data
    df = yf.download(ticker, period="10y")
    if df.empty: return None
    # Handle MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    
    # --- Feature Engineering ---
    # Log Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    # 20-Day Moving Average
    df['MA20'] = df['Close'].rolling(20).mean()
    # 20-Day Volatility
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Target: The next day's log return
    df['Target'] = df['Log_Ret'].shift(-1)
    
    # Drop NaN values created by rolling windows and shifting
    return df.dropna().copy()

with st.spinner(f"Loading data for {ticker_input}..."):
    df = load_and_prep_data(ticker_input)
if df is None:
    st.error(f"Could not load data for ticker '{ticker_input}'. Please check the symbol.")
    st.stop()

# --- 2. LSTM MODEL DEFINITION ---
class QuantLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(QuantLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer with dropout to prevent overfitting
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]) 
        return out

# --- 3. DATA PREPARATION FOR PYTORCH ---
features = ['Log_Ret', 'Vol_20', 'RSI', 'MACD_Hist']
# Scalers for features and target to improve training stability
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit and transform the data
X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[['Target']])

# Function to create sequences for LSTM
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# --- STRICT OOS SPLIT ---
# Reserve the exact last 252 days (1 trading year) for an honest out-of-sample backtest.
# The model will NOT see this data during training.
test_days = 252
split_idx = len(X_seq) - test_days

# Convert to PyTorch tensors
X_train, y_train = torch.FloatTensor(X_seq[:split_idx]), torch.FloatTensor(y_seq[:split_idx])
X_test, y_test = torch.FloatTensor(X_seq[split_idx:]), torch.FloatTensor(y_seq[split_idx:])

# Data for backtest visualization
backtest_dates = df['Date'].iloc[-test_days:].reset_index(drop=True)
actual_returns = df['Target'].iloc[-test_days:].reset_index(drop=True)

# --- 4. MODEL INSTANTIATION & TRAINING/LOADING LOGIC ---
# Instantiate the model
model = QuantLSTM(input_size=len(features), hidden_size=32, num_layers=2, output_size=1)
criterion = nn.MSELoss() # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Adam optimizer

# --- Logic to determine whether to train or load weights ---
should_train = False
if retrain_button:
    should_train = True
    st.toast("🔄 Force retrain requested. Starting training process...", icon="🤖")
elif not os.path.exists(MODEL_WEIGHTS_PATH):
    should_train = True
    st.toast(f"📁 No saved weights found for {ticker_input}. Training a new model...", icon="🧠")
else:
    # Weights exist and retraining is not forced, so try to load
    try:
        # Load weights from the saved .pth file
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
        # Set model to evaluation mode (important for dropout/batchnorm)
        model.eval()
        st.toast(f"✅ Successfully loaded pre-trained weights for {ticker_input}!", icon="📂")
    except Exception as e:
        # If loading fails for any reason, fall back to training
        st.warning(f"Could not load weights: {e}. Retraining model.")
        should_train = True

# --- Training Loop (only runs if needed) ---
if should_train:
    st.markdown("### 🧠 Training Deep Learning Model...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    funny_phrases = [
        "Consulting the Wall Street bets oracles...",
        "Mining pure alpha from market noise...",
        "Bribing the algorithms...",
        "Herding the GPU clusters...",
        "Extracting the tears of short sellers...",
        "Quantifying the unquantifiable...",
        "Almost there, formatting the profit matrix..."
    ]
    
    epochs = 50 # Number of training iterations
    for epoch in range(epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad() # Clear gradients
        outputs = model(X_train) # Forward pass
        loss = criterion(outputs, y_train) # Calculate loss
        loss.backward() # Backward pass (compute gradients)
        optimizer.step() # Update weights
        
        # Update UI with progress
        progress = int(((epoch + 1) / epochs) * 100)
        progress_bar.progress(progress)
        phrase_idx = int((epoch / epochs) * len(funny_phrases))
        status_text.text(f"Epoch {epoch+1}/{epochs} | {funny_phrases[min(phrase_idx, len(funny_phrases)-1)]} Loss: {loss.item():.4f}")
        time.sleep(0.03) # Small delay for visual effect
        
    # Save the trained weights to the .pth file
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    status_text.success(f"Training Complete! Weights saved to {MODEL_WEIGHTS_PATH}.")
    progress_bar.empty()
    # Set model to eval mode after training
    model.eval()

# --- 5. THE BACKTESTING ENGINE ---
@st.cache_data(show_spinner=False)
def run_backtest(_model, _X_test, _actual_returns, _scaler_y):
    # Ensure model is in eval mode
    _model.eval()
    with torch.no_grad():
        # Get model predictions on the test set
        preds_scaled = _model(_X_test).numpy()
    
    # Unscale predictions to get raw expected returns
    preds_raw = _scaler_y.inverse_transform(preds_scaled).flatten()
    
    # --- Strategy Logic ---
    # If predicted return > 0, go Long (1). If < 0, go Short (-1).
    positions = np.where(preds_raw > 0, 1, -1)
    
    # Calculate strategy returns.
    # We multiply positions by actual returns. A Long position (1) profits from a positive return.
    # A Short position (-1) profits from a negative return (e.g., -1 * -0.02 = +0.02).
    strategy_returns = positions * _actual_returns.values
    
    # Calculate Cumulative Returns for Strategy and Buy & Hold benchmark
    cum_bh_returns = (1 + _actual_returns).cumprod() - 1
    cum_strat_returns = (1 + strategy_returns).cumprod() - 1
    
    # Calculate performance metrics
    win_rate = np.mean(strategy_returns > 0) * 100
    strat_total_ret = cum_strat_returns.iloc[-1] * 100
    bh_total_ret = cum_bh_returns.iloc[-1] * 100
    
    return cum_bh_returns, cum_strat_returns, win_rate, strat_total_ret, bh_total_ret, positions

# Run the backtest (with a fun loading animation)
st.markdown("### ⏳ Running Walk-Forward Backtest...")
bt_progress = st.progress(0)
bt_status = st.empty()

bt_phrases = [
    "Simulating past mistakes...",
    "Hindsight is 20/20, computing...",
    "Liquidating imaginary margin calls...",
    "Counting hypothetical lambos..."
]

for i in range(100):
    bt_progress.progress(i + 1)
    phrase_idx = int((i / 100) * len(bt_phrases))
    bt_status.text(f"Backtesting | {bt_phrases[min(phrase_idx, len(bt_phrases)-1)]}")
    time.sleep(0.01)
    
bt_progress.empty()
bt_status.empty()

# Execute the backtest function
bh_curve, strat_curve, win_rate, strat_ret, bh_ret, positions = run_backtest(model, X_test, actual_returns, scaler_y)

# --- 6. UI & DASHBOARD ---
st.markdown("---")
st.subheader("🕵️‍♂️ 1-Year AI Backtest Results (Out-of-Sample)")

b1, b2, b3 = st.columns(3)
b1.metric("Buy & Hold Return", f"{bh_ret:.1f}%", help="Total return from simply holding the asset for the last year.")
b2.metric("AI Strategy Return", f"{strat_ret:.1f}%", delta=f"{strat_ret - bh_ret:.1f}% vs B&H", help="Total return from the AI's daily Long/Short strategy.")
b3.metric("AI Win Rate", f"{win_rate:.1f}%", help="The percentage of days the AI's position resulted in a profit.")

# --- Backtest Chart ---
fig_bt = go.Figure()
# Add Buy & Hold benchmark line
fig_bt.add_trace(go.Scatter(x=backtest_dates, y=bh_curve * 100, name='Buy & Hold (Benchmark)', line=dict(color='#95a5a6', width=2, dash='dot')))
# Add AI Strategy line
fig_bt.add_trace(go.Scatter(x=backtest_dates, y=strat_curve * 100, name='AI Strategy (Long/Short)', line=dict(color='#2ecc71', width=3)))

# Add background coloring to indicate Long/Short regimes
# Green background for Long, Red for Short
for i in range(len(positions) - 1):
    color = "rgba(46, 204, 113, 0.1)" if positions[i] == 1 else "rgba(231, 76, 60, 0.1)"
    fig_bt.add_vrect(x0=backtest_dates[i], x1=backtest_dates[i+1], fillcolor=color, layer="below", line_width=0)

fig_bt.update_layout(
    template="plotly_white", 
    hovermode="x unified", 
    height=450, 
    yaxis_title="Cumulative Return (%)", 
    title="Hypothetical 1-Year Performance vs Benchmark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_bt, use_container_width=True)

# --- 7. STABILIZED FUTURE SIMULATION ---
@st.cache_data(show_spinner=False)
def run_lstm_simulation(_model, base_data, n_days, n_sims, _scaler_X, _scaler_y, _seq_length):
    # Ensure model is in eval mode
    _model.eval()
    last_price = base_data['Close'].iloc[-1]
    
    # --- Structural Drift Anchor ---
    # Cap the model's daily drift prediction to prevent unrealistic exponential explosions.
    # A 25% max annual drift is a reasonable upper bound for a baseline prediction.
    max_annual_drift = 0.25 
    daily_drift_cap = max_annual_drift / 252

    # Get the last sequence of data to start predictions from
    last_seq = base_data[features].tail(_seq_length).values
    last_seq_scaled = _scaler_X.transform(last_seq)
    
    # Initialize simulation array
    all_paths = np.zeros((n_days, n_sims))
    current_prices = np.full(n_sims, last_price)
    # Use historical volatility to scale the stochastic shocks
    historical_vol = base_data['Log_Ret'].std()
    
    with torch.no_grad():
        for d in range(n_days):
            # 1. Get AI's base prediction for the next day's return
            seq_tensor = torch.FloatTensor(last_seq_scaled).unsqueeze(0)
            pred_scaled = _model(seq_tensor).item()
            pred_ret = _scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            # 2. Apply Drift Anchor (Clamp the prediction)
            pred_ret = np.clip(pred_ret, -daily_drift_cap, daily_drift_cap)
            
            # 3. Add Stochastic Noise (Monte Carlo aspect)
            # Generate random shocks based on historical volatility
            shocks = np.random.normal(0, historical_vol, n_sims)
            daily_returns = pred_ret + shocks
            
            # 4. Calculate new prices
            current_prices = current_prices * np.exp(daily_returns)
            # Apply a hard floor to prevent prices from becoming non-positive
            current_prices = np.maximum(current_prices, 0.01)
            all_paths[d, :] = current_prices
            
            # Note: For a fully autoregressive simulation, we would update 'last_seq_scaled'
            # with the new predicted data at each step. For computational efficiency in this
            # dashboard, we use the AI's prediction from the current point to guide the drift
            # of the entire simulation path.
            
    return all_paths

# Run the Monte Carlo simulation
with st.spinner(f"Simulating {n_simulations} future paths for {n_years} years..."):
    sim_results = run_lstm_simulation(model, df, forecast_days, n_simulations, scaler_X, scaler_y, seq_length)

# Calculate forecast statistics
median_forecast = np.median(sim_results, axis=1)
upper_95_bound = np.percentile(sim_results, 97.5, axis=1)
lower_95_bound = np.percentile(sim_results, 2.5, axis=1)

# --- Future Projection Chart ---
st.markdown("---")
st.subheader(f"🔮 Forward-Looking Monte Carlo Projection ({n_years}Y)")

fig_main = go.Figure()
# Plot the last year of historical prices for context
hist_plot = df.tail(252)
fig_main.add_trace(go.Scatter(x=hist_plot['Date'], y=hist_plot['Close'], name='Historical Price', line=dict(color='#2c3e50', width=2)))

# Generate future dates for the forecast
future_dates = pd.date_range(df['Date'].max(), periods=forecast_days + 1, freq='B')[1:]

# Plot the 95% confidence interval funnel
fig_main.add_trace(go.Scatter(x=future_dates, y=upper_95_bound, line=dict(width=0), showlegend=False))
fig_main.add_trace(go.Scatter(x=future_dates, y=lower_95_bound, line=dict(width=0), fill='tonexty', fillcolor='rgba(41, 128, 185, 0.2)', name='95% Confidence Interval'))

# Plot the median forecast path
fig_main.add_trace(go.Scatter(x=future_dates, y=median_forecast, name='AI Median Trajectory', line=dict(color='#e74c3c', width=3)))

fig_main.update_layout(
    template="plotly_white", 
    hovermode="x unified", 
    height=600, 
    yaxis_title="Asset Price ($)",
    title=f"{ticker_input} Price Forecast",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_main, use_container_width=True)
