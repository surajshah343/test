import numpy as np
from typing import Tuple

class HestonMonteCarlo:
    """
    Monte Carlo simulation for Heston stochastic volatility model.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
        r: float
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.r = r
        self.v0 = v0
        self.rho = rho

    def simulate_paths(
        self,
        S0: float,
        T: float,
        steps: int,
        n_paths: int
    ) -> np.ndarray:
        dt = T / steps
        S = np.zeros((n_paths, steps))
        v = np.zeros((n_paths, steps))

        S[:, 0] = S0
        v[:, 0] = self.v0

        for t in range(1, steps):
            z1 = np.random.normal(size=n_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(size=n_paths)

            v[:, t] = np.abs(
                v[:, t-1]
                + self.kappa * (self.theta - v[:, t-1]) * dt
                + self.xi * np.sqrt(v[:, t-1] * dt) * z2
            )

            S[:, t] = (
                S[:, t-1]
                * np.exp((self.r - 0.5 * v[:, t-1]) * dt
                + np.sqrt(v[:, t-1] * dt) * z1)
            )

        return S

    def price_option(
        self,
        S0: float,
        K: float,
        T: float,
        steps: int,
        n_paths: int,
        option_type: str = "call"
    ) -> float:
        paths = self.simulate_paths(S0, T, steps, n_paths)
        ST = paths[:, -1]

        if option_type == "call":
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)

        return np.exp(-self.r * T) * np.mean(payoff)
