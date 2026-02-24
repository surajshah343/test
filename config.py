from dataclasses import dataclass

@dataclass
class ModelConfig:
    lookback: int = 30
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
