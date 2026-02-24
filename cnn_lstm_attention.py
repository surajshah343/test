import torch
import torch.nn as nn
from typing import Tuple

class TemporalAttention(nn.Module):
    """Applies attention across temporal LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class HybridQuantModel(nn.Module):
    """CNN + LSTM + Attention Model"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.attn = TemporalAttention(128)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.relu(self.cnn(x))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attn(lstm_out)
        return self.fc(context)
