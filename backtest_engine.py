import numpy as np
from risk.risk_metrics import RiskMetrics

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
