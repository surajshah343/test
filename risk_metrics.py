import numpy as np

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
