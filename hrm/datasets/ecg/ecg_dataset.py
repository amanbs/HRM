import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGShardDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.idx = []
        for si, f in enumerate(self.files):
            n = np.load(f)["label_ord"].shape[0]
            self.idx += [(si, i) for i in range(n)]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        si, ri = self.idx[i]
        with np.load(self.files[si]) as z:
            tokens = torch.from_numpy(z["tokens"][ri])  # [N,d]
            mask = torch.from_numpy(z["mask"][ri])  # [N]
            yb = torch.tensor(int(z["label_ord"][ri]))
            yr = torch.tensor(float(z["label_reg"][ri]))
        return {
            "tokens": tokens.float(),
            "mask": mask.bool(),
            "label_ord": yb,
            "label_reg": yr,
        }
