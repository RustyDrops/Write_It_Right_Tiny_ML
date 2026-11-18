#!/usr/bin/env python3
"""Write It Right v2: FULL EMNIST byclass train split, small CNN, letters word-mode eval.
Data: EMNIST only (public, NIST SD19 via torchvision). No LLM-generated data/labels/filtering.
- In-memory uint8 arrays, batched affine augmentation (no flips: would corrupt b/d).
- Checkpoint (incl. optimizer/scheduler) after every epoch; --resume to continue.
- Per-epoch quick eval on a fixed 10k test subset (raw + word-mode).
- Final: full official test eval raw 62-class AND word-mode (letters-masked, case-merged),
  ONNX fp32 + dynamic int8 export with v2 filenames (v1 artifacts preserved).
"""
import json, time, random, os, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
torch.set_num_threads(4)
try: torch.set_num_interop_threads(2)
except RuntimeError: pass
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
EPOCHS = 10
BATCH = 512
LR = 2e-3
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
LETTER_IDX = list(range(10, 62))          # class indices that are letters
def letter_id(cls_idx):                  # case-merged letter id 0-25 for a letter class idx
    return (cls_idx - 10) % 26

def load_arrays(train=True):
    ds = datasets.EMNIST(DATA, split="byclass", train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy()   # uint8, stored transposed

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, 62)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

def augment(x):
    """Batched random affine: rot +-8deg, shift +-2px, scale 0.95-1.05. No flips."""
    n = x.size(0)
    ang = (torch.rand(n) * 16 - 8) * math.pi / 180.0
    tx = (torch.rand(n) * 4 - 2) * (2.0 / 28.0)
    ty = (torch.rand(n) * 4 - 2) * (2.0 / 28.0)
    sc = 0.95 + torch.rand(n) * 0.10
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos;  theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin;  theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

@torch.no_grad()
def conf_matrix(model, X, y, limit=None):
    model.eval()
    conf = np.zeros((62, 62), dtype=np.int64)
    n = len(y) if limit is None else min(limit, len(y))
    for i in range(0, n, 1024):
        e = min(i + 1024, n)
        # X stored (N,28,28) transposed; upright = transpose of each image
        imgs = np.transpose(X[i:e], (0, 2, 1))
        xb = torch.from_numpy(np.ascontiguousarray(imgs)).float().unsqueeze(1) / 255.0
        pred = model(xb).argmax(1).numpy()
        for t, p in zip(y[i:e], pred): conf[t, p] += 1
    return conf

@torch.no_grad()
def word_mode_metrics(model, X, y, limit=None):
    """Logits masked to letters only; case-merged. Evaluated on letter-labelled samples."""
    model.eval()
    n = len(y) if limit is None else min(limit, len(y))
    tot = cor = 0
    focus = {ch: [0, 0] for ch in "bdpq"}          # letter-group (both cases): [correct, total]
    focus_lower = {ch: [0, 0] for ch in "bdpq"}    # lowercase-only true labels
    for i in range(0, n, 1024):
        e = min(i + 1024, n)
        imgs = np.transpose(X[i:e], (0, 2, 1))
        xb = torch.from_numpy(np.ascontiguousarray(imgs)).float().unsqueeze(1) / 255.0
        logits = model(xb).numpy()
        yb = y[i:e]
        masked = logits[:, LETTER_IDX]
        pred_cls = np.array(LETTER_IDX)[masked.argmax(1)]
        for t, p in zip(yb, pred_cls):
            if t < 10: continue
            tl, pl = letter_id(int(t)), letter_id(int(p))
            tot += 1; cor += int(tl == pl)
            for ch in "bdpq":
                if tl == letter_id(CLASSES.index(ch)):
                    focus[ch][1] += 1; focus[ch][0] += int(tl == pl)
                if t == CLASSES.index(ch):
                    focus_lower[ch][1] += 1; focus_lower[ch][0] += int(tl == pl)
    out = {"letters_wordmode_acc": float(cor / max(tot, 1)), "letters_n": int(tot),
           "focus_wordmode": {ch: {"acc": float(c / max(t_, 1)), "n": int(t_)} for ch, (c, t_) in focus.items()},
           "focus_wordmode_lowercase_true": {ch: {"acc": float(c / max(t_, 1)), "n": int(t_)} for ch, (c, t_) in focus_lower.items()}}
    return out

def main():
    resume = "--resume" in sys.argv
    stop_epoch = EPOCHS
    for a in sys.argv:
        if a.startswith("--stop-epoch="):
            stop_epoch = int(a.split("=")[1])
    t0 = time.time()
    Xtr, ytr = load_arrays(True); Xte, yte = load_arrays(False)
    print(f"train FULL {len(ytr)} test FULL {len(yte)}", flush=True)
    model = Net(); nparams = sum(p.numel() for p in model.parameters())
    print(f"params {nparams}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(ytr) / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    lossf = nn.CrossEntropyLoss()
    start_ep = 0; start_step = 0
    ckpt_path = os.path.join(HERE, "model_full_last.pt")
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"]; start_step = ck.get("step", 0)
        print(f"resumed at epoch {start_ep} step {start_step}", flush=True)
    CKPT_EVERY = 200  # steps (~100k samples): a VM/session kill loses at most ~15 min
    for ep in range(start_ep, min(EPOCHS, stop_epoch)):
        model.train(); tot = cor = 0
        perm = np.random.RandomState(SEED * 1000 + ep).permutation(len(ytr))  # deterministic per epoch
        nsteps = math.ceil(len(ytr) / BATCH)
        for s in range(nsteps):
            if ep == start_ep and s < start_step: continue
            idx = perm[s*BATCH:(s+1)*BATCH]
            imgs = np.transpose(Xtr[idx], (0, 2, 1))
            x = torch.from_numpy(np.ascontiguousarray(imgs)).float().unsqueeze(1) / 255.0
            x = augment(x)
            yb = torch.from_numpy(ytr[idx].astype(np.int64))
            opt.zero_grad(); out = model(x); loss = lossf(out, yb)
            loss.backward(); opt.step(); sched.step()
            tot += len(yb); cor += (out.argmax(1) == yb).sum().item()
            if (s + 1) % CKPT_EVERY == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "epoch": ep, "step": s + 1}, ckpt_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1, "step": 0}, ckpt_path)
        torch.save(model.state_dict(), os.path.join(HERE, f"model_full_ep{ep+1}.pt"))
        conf10 = conf_matrix(model, Xte, yte, limit=10000)
        raw10 = float(np.trace(conf10) / conf10.sum())
        wm10 = word_mode_metrics(model, Xte, yte, limit=10000)
        print(f"epoch {ep+1}/{EPOCHS} train_acc={cor/tot:.4f} test10k_raw={raw10:.4f} "
              f"test10k_wordmode={wm10['letters_wordmode_acc']:.4f} elapsed={time.time()-t0:.0f}s ckpt saved", flush=True)
    done_epochs = min(EPOCHS, stop_epoch)
    if done_epochs < EPOCHS:
        print(f"CHUNK_DONE stopped after epoch {done_epochs} (resume checkpoint: model_full_last.pt)", flush=True)
        return
    # Final full-test evaluation, both modes
    conf = conf_matrix(model, Xte, yte)
    raw_acc = float(np.trace(conf) / conf.sum())
    wm = word_mode_metrics(model, Xte, yte)
    res = {"version": "v2-full", "overall_test_acc_raw62": raw_acc,
           "params": nparams, "train_samples": int(len(ytr)), "test_samples": int(conf.sum()),
           "epochs": EPOCHS, "word_mode_full": wm, "train_seconds_total_run": time.time() - t0}
    # raw-mode focus (v1-comparable, lowercase true labels, exact class)
    res["focus_raw_lowercase"] = {}
    for ch in "bdpq":
        ci = CLASSES.index(ch); row = conf[ci]; tot_ = int(row.sum())
        res["focus_raw_lowercase"][ch] = {"n": tot_, "acc": float(row[ci] / tot_),
            "confused_with": {CLASSES[j]: float(row[j] / tot_) for j in range(62) if j != ci and row[j] / tot_ >= 0.01}}
    print(json.dumps(res, indent=1), flush=True)
    model.eval()
    onnx_path = os.path.join(HERE, "write_it_right_v2.onnx")
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), onnx_path, input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}}, opset_version=13)
    res["onnx_fp32_bytes"] = os.path.getsize(onnx_path)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qp = os.path.join(HERE, "write_it_right_v2_int8.onnx")
        quantize_dynamic(onnx_path, qp, weight_type=QuantType.QInt8)
        res["onnx_int8_bytes"] = os.path.getsize(qp)
    except Exception as e:
        res["int8_error"] = str(e)[:200]
    with open(os.path.join(HERE, "results_v2.json"), "w") as f: json.dump(res, f, indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
