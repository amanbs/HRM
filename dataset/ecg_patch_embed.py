import torch
import torch.nn as nn


class ECGPatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 12, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(128, d, 3, 2, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,T]
        z = self.net(x)  # [B,d,T']
        return z.transpose(1, 2)  # [B,N(=T'),d]
