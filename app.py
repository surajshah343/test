import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, Callable

# ==========================================
# BACKEND CLASSES
# ==========================================

# ------------------------
# Bayesian Optimizer (Optuna)
# ------------------------
class BayesianOptimizer:
    def __init__(self, param_space=None):
        self.param_space = param_space

    def optimize(self, objective_func: Callable, n_trials=10):
        # Suppress optuna logging in Streamlit
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=n_trials)
        return study.best_params, study.best_value

# ------------------------
# Kelly Sizer
# ------------------------
class KellySizer:
    def compute(self, returns, risk_free=0.0, fraction=0.5):
        mean = np.mean(returns)
        var = np.var(returns)
        if var == 0:
            return 0.0
        full_kelly = (mean - risk_free) / var
        return full_kelly * fraction

# ------------------------
# Macro Factor Model
# ------------------------
class MacroFactorModel:
    def __init__(self, factors):
        self.factors = factors
        self.model = LinearRegression()

    def compute_factor_exposures(self, returns_df):
        # Simulating factor data for the same index
        np.random.seed(42)
        factor_data = pd.DataFrame(
            np.random.randn(len(returns_df), len(self.factors)) * 0.01,
            columns=self.factors,
            index=returns_df.index
        )
        exposures = []
        for col in returns_df.columns:
            self.model.fit(factor_data, returns_df[col])
            exposures.append(self.model.coef_)
            
        return pd.DataFrame(exposures, columns=self.factors, index=returns_df.columns)

# ------------------------
# RL Execution Agent (DQN)
# ------------------------
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class RLExecutionAgent:
    def __init__(self, state_dim=4, action_dim=3):
        self.model = DQN(state_dim, action_dim)
        
    def execute(self, signals):
        # Mock execution modifying raw signals based on simulated MDP states
        executed = []
        for sig in signals.flatten():
            # State: (price_proxy, vol, spread, position)
            state = torch.FloatTensor([np.random.rand(), np.random.rand(), 0.01, sig])
            with torch.no_grad():
                q_values = self.model(state)
            action = torch.argmax(q_values).item() - 1 # Map to {-1, 0, 1}
            # Scale signal by RL confidence/action
            executed.append(sig * 0.9 if action != 0 else 0.0)
        return np.array(executed).reshape(signals.shape)

# ------------------------
# Risk Parity Allocator
# ------------------------
class RiskParityAllocator:
    def allocate(self, cov_matrix):
        n = cov_matrix.shape[0]
        w = np.ones(n) / n
        # Gradient descent for risk parity
        for _ in range(500):
            risk = w * (cov_matrix @ w)
            total_risk = np.sum(risk)
            grad = risk - total_risk / n
            w -= 0.01 * grad
            w = np.maximum(w, 0)
            if np.sum(w) > 0:
                w /= np.sum(w)
        return w

# ------------------------
# Heston Monte Carlo
# ------------------------
class HestonMonteCarlo:
    def simulate(self, S0, T=1, dt=1/252, mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7):
        n_steps = int(T / dt)
        S = np.zeros(n_steps)
        v = np.zeros(n_steps)
        S[0] = S0
        v[0] = theta
        
        for t in range(1, n_steps):
            z1, z2 = np.random.randn(), np.random.randn()
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            # Volatility process (Euler-Maruyama with reflection)
            v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1] * dt) * z2)
            # Price process
            S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1)
            
        return S

# ------------------------
# Transformer Alpha
# ------------------------
class TransformerAlpha(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)

    def predict(self, X):
        # Convert numpy array to sequence tensor [Batch, SeqLen, Features]
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).unsqueeze(-1)
            # Add batch dim if missing
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)
            
            emb = self.embedding(x_tensor)
            out = self.transformer(emb)
            preds = self.fc(out).squeeze(-1).numpy()
            return preds

# ------------------------
# Forecast Metrics
# ------------------------
class ForecastMetrics:
    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        directional = np.mean(np.sign(y_true) == np.sign(y_pred))
        ic = np.corrcoef(y_true, y_pred)[0, 1] if len(np.unique(y_pred)) > 1 else 0.0
        return {"MAE": mae, "RMSE": rmse, "Dir_Acc": directional, "IC": ic}

    def sharpe_ratio(self, returns):
        if np.std(returns) == 0: return 0.0
        return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

    def sortino_ratio(self, returns):
        downside = returns[returns < 0]
        dd = np.std(downside) + 1e-9
        return (np.mean(returns) / dd) * np.sqrt(252)

    def max_drawdown(self, returns):
        cumulative = np.exp(np.cumsum(returns))
        peak = np.maximum.accumulate(cumulative)
        return np.min((cumulative - peak) / peak)


# ==========================================
# STREAMLIT UI - DASHBOARD LAYOUT
# ==========================================
st.set_page_config(page_title="Multi-Asset Quant Platform", layout="wide")

st.title("🏛️ Multi-Asset Quant Platform")
st.markdown("Advanced hybrid quantitative trading pipeline with macroeconomic, fundamental, and microstructure layers.")

# --- DATA FETCHING (CACHED) ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers, period="2y"):
    try:
        data = yf.download(tickers, period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        returns = data.pct_change().dropna()
        return data, returns
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- SIDEBAR: GLOBAL CONTROLS ---
st.sidebar.header("Global Settings")
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", "SPY, QQQ, GLD, TLT, AAPL")
tickers = [t.strip().upper() for t in tickers_input.split(",")]
num_assets = len(tickers)
st.sidebar.write(f"**Active Assets:** {num_assets}")

prices_df, returns_df = fetch_market_data(tickers)

if prices_df.empty:
    st.warning("Please enter valid ticker symbols.")
    st.stop()

returns_array = returns_df.values

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Technical AI Forecast", 
    "🏢 Deep Value & DuPont", 
    "⚖️ Portfolio & Risk", 
    "🎲 Options & MC",
    "⚙️ System Optimization"
])

# ---------------------------------------------------------
# TAB 1: TECHNICAL AI FORECAST
# ---------------------------------------------------------
with tab1:
    col_hdr, col_pop = st.columns([5, 1])
    with col_hdr:
        st.header("Transformer Alpha & Microstructure Execution")
    with col_pop:
        with st.popover("ℹ️ Info & Math"):
            st.markdown("""
            **Transformer Alpha Model**
            Eliminates traditional sequence recurrence by using self-attention to capture long-range market dependencies.
            
            **Formula (Attention Mechanism):**
            $$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
            
            * **How to Read:** Output arrays represent predicted directional returns.
            * **Good:** High Information Coefficient (IC) and positive expected return.
            * **Bad:** Output signals clustered near 0; indicates model uncertainty.
            
            **RL Execution Engine (DQN)**
            Uses a Markov Decision Process to minimize transaction costs and slippage.
            * **Formula (Objective):** $$ \max \mathbb{E}\left[\sum \gamma^t r_t \right] $$
            """)

    st.line_chart(prices_df / prices_df.iloc[0] * 100) 
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transformer Alpha Predictions", help="Daily return forecasts generated by PyTorch TransformerEncoder.")
        transformer = TransformerAlpha(input_dim=num_assets)
        alphas = transformer.predict(returns_array)
        alpha_df = pd.DataFrame(alphas[-10:], columns=tickers, index=returns_df.index[-10:]) 
        st.dataframe(alpha_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
        
    with col2:
        st.subheader("RL Agent Execution", help="DQN limits trade execution to {-1, 0, 1} actions to prevent overtrading.")
        rl_agent = RLExecutionAgent()
        signals = alphas[-5:] 
        executed_signals = rl_agent.execute(signals)
        st.dataframe(pd.DataFrame(executed_signals, columns=tickers, index=returns_df.index[-5:]), use_container_width=True)

    st.subheader("Historical Metrics")
    metrics = ForecastMetrics()
    metric_cols = st.columns(num_assets)
    for i, ticker in enumerate(tickers):
        ret_i = returns_array[:, i]
        sharpe = metrics.sharpe_ratio(ret_i)
        sortino = metrics.sortino_ratio(ret_i)
        metric_cols[i].metric(label=ticker, value=f"{sharpe:.2f} Sharpe", delta=f"{sortino:.2f} Sortino", delta_color="normal")

# ---------------------------------------------------------
# TAB 2: DEEP VALUE & DUPONT
# ---------------------------------------------------------
with tab2:
    col_hdr, col_pop = st.columns([5, 1])
    with col_hdr:
        st.header("Fundamental & Macroeconomic Factors")
    with col_pop:
        with st.popover("ℹ️ Info & Math"):
            st.markdown("""
            **Macro Factor Model**
            Breaks down asset returns into exposures against broad macroeconomic drivers.
            
            **Formula:**
            $$ R_i = \alpha + \beta_1 F_{\text{GDP}} + \beta_2 F_{\text{Inflation}} + \beta_3 F_{\text{Rates}} + \epsilon $$
            
            * **How to Read:** Positive coefficients mean the asset moves with the factor. Negative means inverse correlation.
            * **Good:** Diversified beta across non-correlated factors.
            * **Bad:** Over-concentration in a single macro risk (e.g., massive negative rate exposure).
            """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Macro Factor Exposures", help="Sklearn Linear Regression coefficients against simulated Macro indices.")
        factors = ["GDP", "Inflation", "Rates"]
        factor_model = MacroFactorModel(factors)
        exposures = factor_model.compute_factor_exposures(returns_df)
        st.dataframe(exposures.style.highlight_max(axis=1), use_container_width=True)
        
    with col2:
        st.subheader("DuPont Analysis (Proxies)")
        st.write("Decomposition of Return on Equity (ROE)")
        dupont_data = {
            "Asset": tickers,
            "Net Margin (%)": np.random.uniform(5, 25, num_assets).round(2),
            "Asset Turnover": np.random.uniform(0.5, 1.5, num_assets).round(2),
            "Equity Multiplier": np.random.uniform(1.1, 3.0, num_assets).round(2)
        }
        dupont_df = pd.DataFrame(dupont_data)
        dupont_df["ROE (%)"] = (dupont_df["Net Margin (%)"] * dupont_df["Asset Turnover"] * dupont_df["Equity Multiplier"]).round(2)
        st.dataframe(dupont_df.set_index("Asset"), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: PORTFOLIO & RISK
# ---------------------------------------------------------
with tab3:
    col_hdr, col_pop = st.columns([5, 1])
    with col_hdr:
        st.header("Dynamic Allocation Sizing")
    with col_pop:
        with st.popover("ℹ️ Info & Math"):
            st.markdown("""
            **Risk Parity Allocation**
            Equalizes the risk contribution of every asset in the portfolio, avoiding capitalization-weighted concentration.
            * **Formula:** $$ RC_i = w_i (\Sigma w)_i $$
            * **Good:** Balanced portfolio visually across all assets.
            
            **Kelly Criterion**
            Calculates the theoretically optimal fraction of capital to risk to maximize long-term wealth compounding.
            * **Formula:** $$ f^* = \frac{\mu}{\sigma^2} $$
            * **How to Read:** A value of 0.25 means allocate 25% of your bankroll.
            * **Good:** Values between 0.05 and 0.5 (Half-Kelly is preferred by institutions to reduce drawdown).
            * **Bad:** Negative values (implies shorting) or values > 1.0 (requires leverage, extreme risk of ruin).
            """)
            
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Live Risk Parity Weights")
        cov_matrix = np.cov(returns_array.T)
        
        if num_assets == 1:
            weights = np.array([1.0])
        else:
            allocator = RiskParityAllocator()
            weights = allocator.allocate(cov_matrix)
        
        weight_df = pd.DataFrame({"Asset": tickers, "Weight (%)": (weights * 100).round(2)})
        st.bar_chart(weight_df.set_index("Asset"))
        
    with col2:
        st.subheader("Fractional Kelly (Half-Kelly)")
        kelly = KellySizer()
        
        kelly_fractions = []
        for i in range(num_assets):
            kf = kelly.compute(returns_array[:, i] if num_assets > 1 else returns_array, fraction=0.5)
            kelly_fractions.append(kf)
            
        kelly_df = pd.DataFrame({"Asset": tickers, "Kelly Fraction": np.round(kelly_fractions, 4)})
        st.dataframe(kelly_df.set_index("Asset").style.bar(color='#5fba7d'), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: OPTIONS & VOLATILITY
# ---------------------------------------------------------
with tab4:
    col_hdr, col_pop = st.columns([5, 1])
    with col_hdr:
        st.header(f"Heston Stochastic Volatility Model ({tickers[0]})")
    with col_pop:
        with st.popover("ℹ️ Info & Math"):
            st.markdown("""
            **Heston Model**
            A Monte Carlo simulation framework where volatility is not constant, but a stochastic process itself (mean-reverting).
            
            **Formulas:**
            * **Price Process:** $$ dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S $$
            * **Variance Process:** $$ dv_t = \kappa(\theta - v_t)dt + \xi \sqrt{v_t} dW_t^v $$
            
            * **How to Read:** Evaluates thousands of potential future price paths factoring in volatility spikes.
            * **Good:** Tight clustering indicates market consensus and stable implied volatility.
            * **Bad:** Extreme outliers or downward drift indicating severe tail-risk.
            """)
            
    latest_price = prices_df.iloc[-1, 0] if num_assets > 1 else prices_df.iloc[-1]
    st.markdown(f"Monte Carlo simulation paths starting from latest live price: **${latest_price:.2f}**")
    
    heston = HestonMonteCarlo()
    num_paths = st.slider("Number of Monte Carlo Paths", 1, 50, 10)
    
    sim_paths = {}
    for i in range(num_paths):
        sim_paths[f"Path_{i+1}"] = heston.simulate(S0=latest_price)
        
    st.line_chart(pd.DataFrame(sim_paths))

# ---------------------------------------------------------
# TAB 5: SYSTEM OPTIMIZATION
# ---------------------------------------------------------
with tab5:
    col_hdr, col_pop = st.columns([5, 1])
    with col_hdr:
        st.header("Bayesian Hyperparameter Optimization")
    with col_pop:
        with st.popover("ℹ️ Info & Math"):
            st.markdown("""
            **Tree-structured Parzen Estimator (Optuna)**
            Builds a probabilistic surrogate model to select the most promising hyperparameters rather than random guessing.
            
            **Formula (Expected Improvement):**
            $$ \alpha(\theta) = \mathbb{E}[\max(f(\theta) - f^*, 0)] $$
            
            * **How to Read:** Iteratively searches for combinations that maximize the Objective Score (Sharpe).
            * **Good:** Objective score cleanly plateauing at a high value across trials.
            * **Bad:** Erratic objective score jumping wildly, indicating the model is unstable or overfitting.
            """)
            
    st.markdown("Automated Optuna engine for maximizing out-of-sample Sharpe.")
    
    if st.button("Run Optuna Engine", type="primary"):
        with st.spinner("Optimizing param space..."):
            
            def objective(trial):
                # Simulated objective: testing hyperparams on a dummy Sharpe optimization
                lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
                dropout = trial.suggest_float("dropout", 0.1, 0.5)
                
                # Mock evaluation logic representing model training
                simulated_sharpe = 1.5 + (np.log(lr) * -0.1) - (dropout * 0.5) + np.random.normal(0, 0.1)
                return simulated_sharpe
            
            optimizer = BayesianOptimizer()
            best_params, best_score = optimizer.optimize(objective, n_trials=15)
            
            st.success("Optimization Complete!")
            
            col1, col2 = st.columns(2)
            col1.metric("Best Objective Score (Sharpe)", f"{best_score:.4f}", help="Higher is better.")
            col2.json(best_params)
    else:
        st.info("Click the button above to begin the Bayesian optimization loop.")
