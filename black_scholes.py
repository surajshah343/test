import numpy as np
import scipy.stats as stats

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
