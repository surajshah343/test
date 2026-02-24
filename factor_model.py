import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class MacroFactorModel:

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, asset_returns: pd.Series, factor_returns: pd.DataFrame):
        self.model.fit(factor_returns, asset_returns)
        return self.model.coef_

    def predict(self, factor_returns: pd.DataFrame):
        return self.model.predict(factor_returns)
