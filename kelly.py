import numpy as np

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
