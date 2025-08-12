import torch
from ..models.heads.ef_head import EFHead, EFLoss


class ECG_EF_Task:
    def __init__(self, d_model: int, bins, mono_margin: float = 0.0):
        self.head = EFHead(d_model, K=len(bins) + 1)
        self.loss = EFLoss(K=len(bins) + 1, mono_margin=mono_margin)

    def forward(self, H_states):  # list of [B,d] per cycle
        ords, regs = [], []
        for h in H_states:
            o, r = self.head(h)
            ords.append(o)
            regs.append(r)
        return ords, regs

    def compute_loss(self, outs, batch):
        ords, regs = outs
        return self.loss(ords, regs, batch["label_ord"], batch["label_reg"])
