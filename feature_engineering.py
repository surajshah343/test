import numpy as np
import pandas as pd

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
