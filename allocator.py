import numpy as np
from typing import Tuple

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
