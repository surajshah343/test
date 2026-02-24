import pandas as pd
import numpy as np

class MacroRegime:
    """Simple macro regime classifier via SPY SMA filter."""

    @staticmethod
    def classify(df: pd.DataFrame) -> pd.Series:
        df["SMA_200"] = df["Close"].rolling(200).mean()
        regime = np.where(df["Close"] > df["SMA_200"], 1, -1)
        return pd.Series(regime, index=df.index)
