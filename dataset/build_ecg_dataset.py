import os
import csv
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ecg_patch_embed import ECGPatchEmbed


class _RawECG(Dataset):
    def __init__(self, csv_path: str):
        rows = [r for r in csv.DictReader(open(csv_path))]
        self.items = [
            (r["path"], float(r["ef"]), int(r["ef_bin"])) for r in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, ef, efb = self.items[i]
        x = np.load(p)  # [C,T], T≈2500 for 10s
        return (
            torch.from_numpy(x).float(),
            torch.tensor(ef).float(),
            torch.tensor(efb).long(),
        )


def build_shards(
    csv_path: str,
    out_dir: str,
    shard_size: int = 10000,
    in_ch: int = 12,
    d: int = 256,
):
    os.makedirs(out_dir, exist_ok=True)
    ds = _RawECG(csv_path)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    embed = ECGPatchEmbed(in_ch=in_ch, d=d).eval()
    with torch.no_grad():
        buf_tokens, buf_masks, buf_efo, buf_efr = [], [], [], []
        sid = 0
        n = 0
        for xb, efr, efo in dl:
            xb = xb  # CPU OK; switch to cuda() if large
            tokens = embed(xb).cpu()  # [B,N,d]
            B, N, D = tokens.shape
            mask = torch.zeros(B, N, dtype=torch.bool)
            buf_tokens.append(tokens.numpy())
            buf_masks.append(mask.numpy())
            buf_efo.append(efo.numpy())
            buf_efr.append(efr.numpy())
            n += B
            if n >= shard_size:
                np.savez_compressed(
                    os.path.join(out_dir, f"ecg_{sid:05d}.npz"),
                    tokens=np.concatenate(buf_tokens, 0),
                    mask=np.concatenate(buf_masks, 0),
                    label_ord=np.concatenate(buf_efo, 0),
                    label_reg=np.concatenate(buf_efr, 0),
                )
                sid += 1
                n = 0
                buf_tokens, buf_masks, buf_efo, buf_efr = [], [], [], []
        if n > 0:
            np.savez_compressed(
                os.path.join(out_dir, f"ecg_{sid:05d}.npz"),
                tokens=np.concatenate(buf_tokens, 0),
                mask=np.concatenate(buf_masks, 0),
                label_ord=np.concatenate(buf_efo, 0),
                label_reg=np.concatenate(buf_efr, 0),
            )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--in_ch", type=int, default=12)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--shard_size", type=int, default=10000)
    args = ap.parse_args()
    build_shards(args.csv, args.out, args.shard_size, args.in_ch, args.d)
