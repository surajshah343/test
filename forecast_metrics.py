import numpy as np
from typing import Dict

class ForecastMetrics:

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        directional = np.mean(np.sign(y_true) == np.sign(y_pred))
        ic = np.corrcoef(y_true, y_pred)[0, 1]

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Directional Accuracy": float(directional),
            "Information Coefficient": float(ic)
        }
