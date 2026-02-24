import optuna
import numpy as np
from typing import Callable

class BayesianOptimizer:
    """
    Bayesian optimization for model hyperparameters.
    """

    def __init__(self, objective_fn: Callable):
        self.objective_fn = objective_fn

    def optimize(self, n_trials: int = 50):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_fn, n_trials=n_trials)
        return study.best_params, study.best_value
    def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    sharpe = train_and_evaluate(lr, hidden_dim, dropout)
    return sharpe
