import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import warnings
from typing import Tuple, List, Dict, Optional, Any

warnings.filterwarnings('ignore')

# --- CONFIGURATION & HARDWARE ---
st.set_page_config(page_title="Institutional Quant Engine", layout="wide", initial_sidebar_state="expanded")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# ==========================================
# 1. CORE NEURAL ARCHITECTURE
# ==========================================
class TemporalAttention(nn.Module):
    """Attention mechanism to weigh temporal importance of LSTM outputs."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector, attn_weights

class HybridQuantModel(nn.Module):
    """CNN for spatial feature extraction, LSTM for temporal sequencing, Attention for weighting."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.attention = TemporalAttention(128)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # Conv1d expects (Batch, Channels, Length)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # Back to (Batch, Length, Channels) for LSTM
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)

# ==========================================
# 2. DATA INGESTION & FEATURE ENGINEERING
# ==========================================
class QuantDataPipeline:
    """Handles data retrieval, institutional feature engineering, and macro regime filtering."""
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_macro_regime() -> pd.DataFrame:
        """Fetches SPY data to determine the broader market regime."""
        spy = yf.download("SPY", start="2015-01-01", interval="1d", progress=False)
        if isinstance(spy.columns, pd.MultiIndex): 
            spy.columns = spy.columns.get_level_values(0)
        spy['SMA_200'] = spy['Close'].rolling(window=200).mean()
        spy['Regime'] = np.where(spy['Close'] > spy['SMA_200'], 'Bull', 'Bear')
        return spy.reset_index()[['Date', 'Regime']]

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_and_process(symbol: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(symbol, start="2015-01-01", interval="1d", progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()

            # Base Technicals
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Vol_20'] = df['Log_Ret'].rolling(20).std()
            df['RSI'] = QuantDataPipeline._calculate_rsi(df['Close'], 14)
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Smart Money Indicators
            df['VWMA_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-9)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()

            # Negative Volume Index (NVI)
            nvi_vals = [1000]
            for i in range(1, len(df)):
                if df['Volume'].iloc[i] < df['Volume'].iloc[i-1]:
                    ret = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                    nvi_vals.append(nvi_vals[-1] * (1 + ret))
                else:
                    nvi_vals.append(nvi_vals[-1])
            df['NVI'] = nvi_vals
            df['NVI_Signal'] = df['NVI'].ewm(span=255, adjust=False).mean()

            # Merge Macro Regime
            macro = QuantDataPipeline.fetch_macro_regime()
            df['Date'] = pd.to_datetime(df['Date'])
            macro['Date'] = pd.to_datetime(macro['Date'])
            df = pd.merge(df, macro, on='Date', how='left').fillna(method='ffill')

            return df.dropna().reset_index(drop=True)
        except Exception as e:
            st.error(f"Data Pipeline Error: {e}")
            return None

# ==========================================
# 3. BACKTESTING & RISK ENGINE
# ==========================================
class RiskEngine:
    """Calculates institutional-grade risk metrics."""
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        if len(returns) == 0: return 0.0
        cumulative = np.exp(np.cumsum(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-9)
        return np.min(drawdowns) * 100

    @staticmethod
    def calculate_sortino(returns: np.ndarray, rf_rate: float = 0.0) -> float:
        if len(returns) == 0: return 0.0
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
        expected_return = np.mean(returns) * 252 - rf_rate
        return expected_return / (downside_std * np.sqrt(252) + 1e-9)

    @staticmethod
    def calculate_var_95(returns: np.ndarray) -> float:
        """Historical Value at Risk (95% confidence)."""
        if len(returns) == 0: return 0.0
        return np.percentile(returns, 5) * 100

class MLBacktester:
    """Handles Purged K-Fold Cross Validation to prevent data leakage."""
    
    @staticmethod
    def create_sequences(data: pd.DataFrame, features: List[str], target_col: str, 
                         s_x: RobustScaler, s_y: RobustScaler, window: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sc = s_x.transform(data[features])
        y_sc = s_y.transform(data[[target_col]])
        xs, ys = [], []
        for i in range(len(x_sc) - window):
            xs.append(x_sc[i:i+window])
            ys.append(y_sc[i+window])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

    @staticmethod
    def purged_time_series_split(df: pd.DataFrame, n_splits: int, purge_gap: int = 20):
        """Yields train/test indices with an embargo/purge gap to prevent moving average data leakage."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(df):
            # Apply purge gap
            if len(train_idx) > purge_gap:
                train_idx = train_idx[:-purge_gap]
            yield train_idx, test_idx

# ==========================================
# 4. DASHBOARD UI
# ==========================================
def main():
    st.sidebar.header("🕹️ Institutional Quant Engine")
    ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()
    lookback = st.sidebar.slider("Lookback Window:", 10, 60, 30)
    epochs = st.sidebar.slider("Training Epochs:", 10, 100, 30)
    n_splits = st.sidebar.slider("Purged K-Folds:", 2, 5, 3)
    forecast_horizon = st.sidebar.selectbox("Horizon:", ["1 Week", "1 Month"])
    
    df = QuantDataPipeline.load_and_process(ticker)

    if df is not None:
        current_regime = df['Regime'].iloc[-1]
        regime_color = "🟢" if current_regime == "Bull" else "🔴"
        st.title(f"🏛️ Quant Dashboard: {ticker} | Macro Regime: {regime_color} {current_regime}")
        
        tab1, tab2 = st.tabs(["📉 Backtest & AI Forecast", "🔬 Institutional Technicals"])

        with tab1:
            st.subheader("Model Training via Purged K-Fold Cross Validation")
            if st.button("🔄 Execute Backtest & Forecast Pipeline"):
                features = ['Log_Ret', 'Vol_20', 'RSI', 'VWMA_20']
                target_col = 'Log_Ret'
                
                y_actual_all, y_pred_ai_all = [], []
                strat_rets_ai_all = []
                last_train_df, last_test_df = None, None

                with st.status("Training Hybrid CNN-LSTM-Attention Model...", expanded=True) as status:
                    # Implement Purged Walk-Forward CV
                    splits = list(MLBacktester.purged_time_series_split(df, n_splits=n_splits, purge_gap=lookback))
                    for i, (train_idx, test_idx) in enumerate(splits):
                        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
                        if i == n_splits - 1: last_train_df, last_test_df = train_df, test_df

                        sc_x = RobustScaler().fit(train_df[features])
                        sc_y = RobustScaler().fit(train_df[[target_col]])
                        
                        X_train, y_train = MLBacktester.create_sequences(train_df, features, target_col, sc_x, sc_y, lookback)
                        X_test, y_test = MLBacktester.create_sequences(test_df, features, target_col, sc_x, sc_y, lookback)
                        
                        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
                        
                        model = HybridQuantModel(len(features)).to(DEVICE)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                        criterion = nn.HuberLoss()
                        
                        for epoch in range(epochs):
                            model.train()
                            for batch_x, batch_y in train_loader:
                                optimizer.zero_grad()
                                loss = criterion(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
                                loss.backward()
                                optimizer.step()
                        
                        model.eval()
                        with torch.no_grad():
                            preds_scaled = model(X_test.to(DEVICE)).cpu().numpy()
                            y_pred = sc_y.inverse_transform(preds_scaled).flatten()
                            y_act = sc_y.inverse_transform(y_test.numpy()).flatten()
                            
                            y_actual_all.extend(y_act)
                            y_pred_ai_all.extend(y_pred)
                            
                            # Regime adjusted position sizing
                            regime_multiplier = np.where(test_df['Regime'].iloc[lookback:] == 'Bull', 1.0, 0.5)
                            ai_rets = np.where(y_pred > 0, 1, -1) * y_act * regime_multiplier
                            strat_rets_ai_all.extend(ai_rets)

                    status.update(label="Validation Complete", state="complete")

                # Risk Metrics Output
                rets_arr = np.array(strat_rets_ai_all)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Cumulative Return", f"{np.sum(rets_arr)*100:.2f}%")
                col2.metric("Sortino Ratio", f"{RiskEngine.calculate_sortino(rets_arr):.2f}")
                col3.metric("Max Drawdown", f"{RiskEngine.calculate_max_drawdown(rets_arr):.2f}%")
                col4.metric("95% VaR (Daily)", f"{RiskEngine.calculate_var_95(rets_arr):.2f}%")

        with tab2:
            st.subheader("Smart Money & Momentum Overlay")
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05)
            
            # Price & VWMA
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price', line=dict(color='white')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['VWMA_20'], name='VWMA', line=dict(color='yellow')), row=1, col=1)
            
            # NVI (Smart Money)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['NVI'], name='NVI', line=dict(color='purple')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['NVI_Signal'], name='NVI Signal', line=dict(dash='dot', color='gray')), row=2, col=1)
            
            # OBV
            fig.add_trace(go.Scatter(x=df['Date'], y=df['OBV'], name='OBV', line=dict(color='cyan')), row=3, col=1)

            fig.update_layout(template="plotly_dark", height=800, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
