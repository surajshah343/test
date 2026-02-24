# app.py
import streamlit as st
import numpy as np
import pandas as pd

# ------------------------
# Bayesian Optimizer
# ------------------------
class BayesianOptimizer:
    def __init__(self, param_space):
        self.param_space = param_space

    def optimize(self, objective_func, n_iter=10):
        best_score = -np.inf
        best_params = None
        for _ in range(n_iter):
            sample = {k: np.random.uniform(*v) for k, v in self.param_space.items()}
            score = objective_func(sample)
            if score > best_score:
                best_score = score
                best_params = sample
        return best_params, best_score

# ------------------------
# Kelly Sizer
# ------------------------
class KellySizer:
    def compute(self, returns, risk_free=0.0):
        mean = np.mean(returns)
        var = np.var(returns)
        if var == 0:
            return 0.0
        return (mean - risk_free) / var

# ------------------------
# Macro Factor Model
# ------------------------
class MacroFactorModel:
    def __init__(self, factors):
        self.factors = factors

    def compute_factor_exposures(self, returns):
        exposures = pd.DataFrame(np.random.randn(*returns.shape), columns=self.factors)
        return exposures

# ------------------------
# RL Execution Agent (dummy example)
# ------------------------
class RLExecutionAgent:
    def execute(self, signals):
        # For demo, just scale signals
        return signals * 0.9

# ------------------------
# Risk Parity Allocator
# ------------------------
class RiskParityAllocator:
    def allocate(self, cov_matrix):
        inv_risk = 1 / np.diag(cov_matrix)
        weights = inv_risk / inv_risk.sum()
        return weights

# ------------------------
# Heston Monte Carlo (simplified)
# ------------------------
class HestonMonteCarlo:
    def simulate(self, S0, T=1, dt=1/252, mu=0.05, kappa=2, theta=0.04, sigma=0.3, rho=-0.7):
        n_steps = int(T / dt)
        S = np.zeros(n_steps)
        S[0] = S0
        v = theta
        for t in range(1, n_steps):
            z1, z2 = np.random.randn(), np.random.randn()
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            v = np.abs(v + kappa*(theta - v)*dt + sigma*np.sqrt(v*dt)*z2)
            S[t] = S[t-1]*np.exp((mu - 0.5*v)*dt + np.sqrt(v*dt)*z1)
        return S

# ------------------------
# Transformer Alpha (dummy)
# ------------------------
class TransformerAlpha:
    def predict(self, X):
        # Just return random alpha predictions
        return np.random.randn(*X.shape)

# ------------------------
# Forecast Metrics
# ------------------------
class ForecastMetrics:
    def sharpe_ratio(self, returns):
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)

# ------------------------
# Streamlit UI
# ------------------------
st.title("Multi-Asset Quant Platform")

num_assets = st.number_input("Select number of assets", min_value=1, max_value=20, value=5)
st.write(f"Selected {num_assets} assets")

# Simulated returns
returns = np.random.randn(100, num_assets)

# Bayesian optimization example
param_space = {"lr": (0.001, 0.1), "gamma": (0.8, 0.99)}
optimizer = BayesianOptimizer(param_space)

def dummy_objective(params):
    return np.random.rand()

best_params, best_score = optimizer.optimize(dummy_objective, n_iter=5)
st.write("Bayesian optimization result:", best_params, "Score:", best_score)

# Kelly sizing
kelly = KellySizer()
kelly_fractions = [kelly.compute(returns[:, i]) for i in range(num_assets)]
st.write("Kelly-optimal fractions:", kelly_fractions)

# Factor exposures
factors = ["GDP", "Inflation", "Rates"]
factor_model = MacroFactorModel(factors)
exposures = factor_model.compute_factor_exposures(pd.DataFrame(returns))
st.write("Macro factor exposures:", exposures.head())

# RL execution signals
rl_agent = RLExecutionAgent()
signals = np.random.randn(100, num_assets)
executed_signals = rl_agent.execute(signals)
st.write("RL executed signals (sample):", executed_signals[:5])

# Risk parity allocation
cov_matrix = np.cov(returns.T)
allocator = RiskParityAllocator()
weights = allocator.allocate(cov_matrix)
st.write("Risk parity weights:", weights)

# Heston Monte Carlo simulation
heston = HestonMonteCarlo()
sim_prices = heston.simulate(S0=100)
st.line_chart(sim_prices)

# Transformer alpha predictions
transformer = TransformerAlpha()
alphas = transformer.predict(returns)
st.write("Transformer alpha predictions (sample):", alphas[:5])

# Forecast metrics
metrics = ForecastMetrics()
sharpe = metrics.sharpe_ratio(returns[:, 0])
st.write(f"Sharpe ratio of first asset: {sharpe:.2f}")
