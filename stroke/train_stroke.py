#!/usr/bin/env python3
"""Tiny stroke-sequence classifier (DESIGN.md frozen architecture):
Conv1d(5->48,k5) BN ReLU MaxPool2 -> Conv1d(48->96,k5) BN ReLU -> GAP ->
FC(96->26). Adam lr1e-3, batch 64, max 60 epochs, early stop on val acc
(patience 10, restore best). Seed 0. Used identically by Arm 1 and Arm 3.

Usage: train_stroke.py <train_npz> <out_prefix> [--val-writer-mod]
  train_npz keys: X_train (N,5,64), y_train, and either w_train (writer ids;
  val = writers with id % 7 == 0) or X_val/y_val given directly.
  Test arrays (X_test, y_test) in the same npz are evaluated once at the end.
Outputs: <out_prefix>_model.pt, <out_prefix>_pred.npz, <out_prefix>_info.json
"""
import os, sys, json, io
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

class StrokeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(5, 48, 5, padding=2)
        self.b1 = nn.BatchNorm1d(48)
        self.c2 = nn.Conv1d(48, 96, 5, padding=2)
        self.b2 = nn.BatchNorm1d(96)
        self.fc = nn.Linear(96, 26)
    def forward(self, x):
        x = torch.relu(self.b1(self.c1(x)))
        x = torch.max_pool1d(x, 2)
        x = torch.relu(self.b2(self.c2(x)))
        x = x.mean(dim=2)
        return self.fc(x)

def run(npz_path, out_prefix):
    d = np.load(npz_path)
    X, y = d["X_train"], d["y_train"]
    if "X_val" in d:
        Xv, yv = d["X_val"], d["y_val"]
        Xtr, ytr = X, y
    else:
        w = d["w_train"]
        val_mask = (w % 7) == 0
        Xtr, ytr = X[~val_mask], y[~val_mask]
        Xv, yv = X[val_mask], y[val_mask]
    Xte, yte = d["X_test"], d["y_test"]
    print(f"train={len(Xtr)} val={len(Xv)} test={len(Xte)}", flush=True)

    model = StrokeNet()
    nparams = sum(p.numel() for p in model.parameters())
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    info = {"n_params": int(nparams), "fp32_state_dict_bytes": len(buf.getvalue()),
            "int8_estimate_bytes": int(nparams)}
    print("params:", info, flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    Xv_t = torch.from_numpy(Xv); yv = torch.from_numpy(yv)
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(60):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vacc = float((model(Xv_t).argmax(1) == yv).float().mean())
        if vacc > best_acc:
            best_acc, bad = vacc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                print(f"early stop at epoch {ep}", flush=True)
                break
        if ep % 10 == 0 or ep == 59:
            print(f"epoch {ep}: val_acc={vacc:.4f} best={best_acc:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xte_t = torch.from_numpy(Xte)
        pred = np.concatenate([model(Xte_t[i:i + 1024]).argmax(1).numpy()
                               for i in range(0, len(Xte_t), 1024)])
    acc = float((pred == yte).mean())
    print(f"TEST acc: {acc:.4f} (val best {best_acc:.4f})", flush=True)
    torch.save(model.state_dict(), out_prefix + "_model.pt")
    np.savez(out_prefix + "_pred.npz", y_true=yte, y_pred=pred)
    info.update({"val_acc": best_acc, "test_acc": acc, "n_train": int(len(Xtr)),
                 "n_val": int(len(Xv)), "n_test": int(len(Xte))})
    with open(out_prefix + "_info.json", "w") as f:
        json.dump(info, f, indent=1)
    return info

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
