import numpy as np
from typing import Iterator, Tuple

class PurgedKFold:
    """Prevents leakage in overlapping time series splits."""

    def __init__(self, n_splits: int, embargo: int = 5):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size

            train_idx = np.concatenate([
                np.arange(0, max(0, test_start - self.embargo)),
                np.arange(min(n, test_end + self.embargo), n)
            ])
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx
