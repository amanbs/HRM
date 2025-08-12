import torch
import torch.nn as nn


def coral_targets(yb: torch.Tensor, K: int = 4) -> torch.Tensor:
    B = yb.size(0)
    t = torch.arange(K - 1, device=yb.device).unsqueeze(0).expand(B, -1)
    return (yb.unsqueeze(1) > t).float()


class EFHead(nn.Module):
    def __init__(self, d: int, K: int = 4):
        super().__init__()
        self.ord = nn.Linear(d, K - 1)
        self.reg = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor):  # h: [B,d] (H-state)
        return self.ord(h), self.reg(h).squeeze(-1)


class EFLoss(nn.Module):
    def __init__(self, K: int = 4, mono_margin: float = 0.0):
        super().__init__()
        self.K = K
        self.bce = nn.BCEWithLogitsLoss()
        self.hub = nn.SmoothL1Loss()
        self.m = mono_margin

    def forward(self, ord_logits_list, reg_list, yb, yr):
        yord = coral_targets(yb, self.K)
        stage = []
        for ol, rg in zip(ord_logits_list, reg_list):
            stage.append(self.bce(ol, yord) + self.hub(rg, yr))
        mono = 0.0
        for i in range(1, len(stage)):
            mono = mono + torch.clamp(stage[i - 1] - stage[i] - self.m, min=0.0)
        return sum(stage) + mono
