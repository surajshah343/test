import sys
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Iterator, Callable, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import yfinance as yf
import streamlit as st
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class ModelConfig:
    lookback: int = 30
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4

# ==========================================
# 2. DATA & FEATURES
# ==========================================
class DataIngestion:
    """Handles market data ingestion with caching and error handling."""
    @staticmethod
    @lru_cache(maxsize=32)
    def fetch_price_data(symbol: str, start: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(symbol, start=start, progress=False)
            if df.empty:
                return None
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            raise ConnectionError(f"Data fetch failed: {e}")

class FeatureEngineering:
    """Feature pipeline with vectorized operations."""
    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Vol_20"] = df["Log_Ret"].rolling(20).std()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        return df.dropna()

# ==========================================
# 3. CORE AI MODELS
# ==========================================
class TemporalAttention(nn.Module):
    """Applies attention across temporal LSTM outputs."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights

class HybridQuantModel(nn.Module):
    """CNN + LSTM + Attention Model"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.attn = TemporalAttention(128)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.relu(self.cnn(x))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attn(lstm_out)
        return self.fc(context)

class TransformerAlpha(nn.Module):
    """Transformer encoder for financial time series forecasting."""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# ==========================================
# 4. MACRO & OPTIONS ENGINES
# ==========================================
class MacroRegime:
    """Simple macro regime classifier via SPY SMA filter."""
    @staticmethod
    def classify(df: pd.DataFrame) -> pd.Series:
        df["SMA_200"] = df["Close"].rolling(200).mean()
        regime = np.where(df["Close"] > df["SMA_200"], 1, -1)
        return pd.Series(regime, index=df.index)

class MacroFactorModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, asset_returns: pd.Series, factor_returns: pd.DataFrame):
        self.model.fit(factor_returns, asset_returns)
        return self.model.coef_

    def predict(self, factor_returns: pd.DataFrame):
        return self.model.predict(factor_returns)

class BlackScholes:
    @staticmethod
    def price(S, K, T, r, sigma, option_type="call"):
        if T <= 0:
            return max(0.0, S - K)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type == "call":
            return S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
        return K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)

class HestonMonteCarlo:
    """Monte Carlo simulation for Heston stochastic volatility model."""
    def __init__(self, kappa: float, theta: float, xi: float, rho: float, v0: float, r: float) -> None:
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.r = r
        self.v0 = v0
        self.rho = rho

    def simulate_paths(self, S0: float, T: float, steps: int, n_paths: int) -> np.ndarray:
        dt = T / steps
        S = np.zeros((n_paths, steps))
        v = np.zeros((n_paths, steps))
        S[:, 0] = S0
        v[:, 0] = self.v0

        for t in range(1, steps):
            z1 = np.random.normal(size=n_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(size=n_paths)
            v[:, t] = np.abs(v[:, t-1] + self.kappa * (self.theta - v[:, t-1]) * dt + self.xi * np.sqrt(v[:, t-1] * dt) * z2)
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * v[:, t-1]) * dt + np.sqrt(v[:, t-1] * dt) * z1)
        return S

    def price_option(self, S0: float, K: float, T: float, steps: int, n_paths: int, option_type: str = "call") -> float:
        paths = self.simulate_paths(S0, T, steps, n_paths)
        ST = paths[:, -1]
        if option_type == "call":
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        return np.exp(-self.r * T) * np.mean(payoff)

# ==========================================
# 5. RISK, PORTFOLIO & EXECUTION
# ==========================================
class RiskMetrics:
    """Institutional risk metrics."""
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        cumulative = np.exp(np.cumsum(returns))
        peak = np.maximum.accumulate(cumulative)
        return np.min((cumulative - peak) / peak)

    @staticmethod
    def sortino_ratio(returns: np.ndarray) -> float:
        downside = returns[returns < 0]
        dd = np.std(downside) + 1e-9
        return np.mean(returns) / dd * np.sqrt(252)

    @staticmethod
    def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
        return np.percentile(returns, 100 * alpha)

class KellySizer:
    @staticmethod
    def kelly_fraction(returns: np.ndarray) -> float:
        mu = np.mean(returns)
        var = np.var(returns)
        if var <= 0:
            return 0.0
        return mu / var

    @staticmethod
    def fractional_kelly(returns: np.ndarray, fraction: float = 0.5) -> float:
        return fraction * KellySizer.kelly_fraction(returns)

class RiskParityAllocator:
    @staticmethod
    def allocate(cov: np.ndarray) -> np.ndarray:
        n = cov.shape[0]
        w = np.ones(n) / n
        for _ in range(500):
            risk = w * (cov @ w)
            total_risk = np.sum(risk)
            grad = risk - total_risk / n
            w -= 0.01 * grad
            w = np.maximum(w, 0)
            w /= np.sum(w)
        return w

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
    def __init__(self, state_dim: int, action_dim: int):
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

# ==========================================
# 6. VALIDATION & METRICS
# ==========================================
class PurgedKFold:
    """Prevents leakage in overlapping time series splits."""
    def __init__(self, n_splits: int, embargo: int = 5):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size
            train_idx = np.concatenate([
                np.arange(0, max(0, test_start - self.embargo)),
                np.arange(min(n, test_end + self.embargo), n)
            ])
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

class BacktestEngine:
    """Strategy evaluation engine."""
    def __init__(self, returns: np.ndarray):
        self.returns = returns

    def performance_summary(self) -> dict:
        return {
            "Cumulative Return": float(np.sum(self.returns)),
            "Max Drawdown": float(RiskMetrics.max_drawdown(self.returns)),
            "Sortino Ratio": float(RiskMetrics.sortino_ratio(self.returns)),
            "VaR (5%)": float(RiskMetrics.value_at_risk(self.returns))
        }

class ForecastMetrics:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        directional = np.mean(np.sign(y_true) == np.sign(y_pred))
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Directional Accuracy": float(directional),
            "Information Coefficient": float(ic)
        }

class BayesianOptimizer:
    """Bayesian optimization for model hyperparameters."""
    def __init__(self, objective_fn: Callable):
        self.objective_fn = objective_fn

    def optimize(self, n_trials: int = 50):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_fn, n_trials=n_trials)
        return study.best_params, study.best_value

# ==========================================
# 7. STREAMLIT UI LAUNCHER
# ==========================================
def main():
    st.set_page_config(page_title="Institutional Quant Pipeline", layout="wide")
    st.title("Hybrid Quantitative Trading Pipeline")
    st.success("All institutional modules have been successfully loaded from a single file.")
    
    st.sidebar.header("Advanced Controls")
    
    # Placeholder for the Bayesian Optimization trigger
    if st.sidebar.button("Run Bayesian Optimization"):
        st.info("Bayesian Optimizer initialized. (Connect to training loop to execute)")

    st.write("### System Architecture Active")
    st.write("The following modules are now running in memory without import errors:")
    st.code("""
    - Hybrid CNN-LSTM-Attention Model
    - Transformer Alpha Model
    - Heston Stochastic Volatility Engine
    - Macro Factor Model & Regime Filter
    - RL Execution Agent
    - Risk Parity & Kelly Sizing
    - Purged K-Fold Validation
    """)

if __name__ == "__main__":
    main()
