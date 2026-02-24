import yfinance as yf
import pandas as pd
from functools import lru_cache
from typing import Optional

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
