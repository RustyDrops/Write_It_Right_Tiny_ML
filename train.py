#!/usr/bin/env python3
"""Write It Right baseline: EMNIST byclass (62 classes) small CNN -> ONNX.
Data: EMNIST only (public, NIST SD19 via torchvision). No LLM-generated data/labels.
Training subsample: stratified cap per class from the official train split.
Evaluation: FULL official test split.
"""
import json, time, random, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
torch.set_num_threads(4)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
CAP_PER_CLASS = 1500
EPOCHS = 4
BATCH = 256
LR = 2e-3

CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]

class Transposed(Dataset):
    """EMNIST images are stored transposed vs upright; swap H/W so letters are upright."""
    def __init__(self, base, train=False):
        self.base = base; self.train = train
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        x = x.transpose(1, 2)  # (1,28,28) swap H,W
        if self.train:
            ang = random.uniform(-10, 10)
            x = transforms.functional.rotate(x, ang)
            dx, dy = random.randint(-2, 2), random.randint(-2, 2)
            x = transforms.functional.affine(x, 0, [dx, dy], 1.0, 0)
        return x, y

def load(split_train=True):
    tf = transforms.Compose([transforms.ToTensor()])
    return datasets.EMNIST(DATA, split="byclass", train=split_train, download=True, transform=tf)

def stratified_subset(ds, cap):
    labels = np.array(ds.targets)
    idx = []
    for c in range(62):
        ci = np.where(labels == c)[0]
        rng = np.random.RandomState(SEED + c)
        rng.shuffle(ci)
        idx.extend(ci[:cap].tolist())
    random.shuffle(idx)
    return Subset(ds, idx)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 62)
    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)

def evaluate(model, loader):
    model.eval()
    conf = np.zeros((62, 62), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1).numpy()
            for t, p in zip(y.numpy(), pred):
                conf[t, p] += 1
    return conf

def main():
    t0 = time.time()
    train_ds_raw = load(True)
    test_ds_raw = load(False)
    train_ds = Transposed(stratified_subset(train_ds_raw, CAP_PER_CLASS), train=True)
    test_ds = Transposed(test_ds_raw, train=False)
    print(f"train subsample: {len(train_ds)}  test (full official): {len(test_ds)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, num_workers=2)

    model = Net()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params: {nparams}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(train_loader))
    lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train(); tot = cor = 0
        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            loss = lossf(out, y)
            loss.backward(); opt.step(); sched.step()
            tot += len(y); cor += (out.argmax(1) == y).sum().item()
        print(f"epoch {ep+1}/{EPOCHS} train_acc={cor/tot:.4f} elapsed={time.time()-t0:.0f}s", flush=True)

    conf = evaluate(model, test_loader)
    acc = np.trace(conf) / conf.sum()
    letter_idx = list(range(10, 62)); lower_idx = list(range(36, 62))
    letter_acc = conf[np.ix_(letter_idx, letter_idx)].trace() / conf[letter_idx].sum()
    lower_acc = conf[np.ix_(lower_idx, lower_idx)].trace() / conf[lower_idx].sum()
    focus = {}
    for ch in "bdpq":
        ci = CLASSES.index(ch)
        row = conf[ci]; total = row.sum()
        focus[ch] = {"n": int(total), "acc": float(row[ci] / total),
                     "confused_with": {CLASSES[j]: float(row[j] / total) for j in range(62)
                                       if j != ci and row[j] / total >= 0.01}}
    results = {"overall_test_acc": float(acc), "letters_acc": float(letter_acc),
               "lowercase_acc": float(lower_acc), "params": nparams,
               "train_samples": len(train_ds), "test_samples": int(conf.sum()),
               "epochs": EPOCHS, "focus_bdpq": focus,
               "train_seconds": time.time() - t0}
    print(json.dumps(results, indent=1), flush=True)

    # Export ONNX (fp32)
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28)
    onnx_path = os.path.join(HERE, "write_it_right.onnx")
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}}, opset_version=13)
    results["onnx_fp32_bytes"] = os.path.getsize(onnx_path)

    # Try int8 dynamic quantization via onnxruntime if available
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qpath = os.path.join(HERE, "write_it_right_int8.onnx")
        quantize_dynamic(onnx_path, qpath, weight_type=QuantType.QInt8)
        results["onnx_int8_bytes"] = os.path.getsize(qpath)
    except Exception as e:
        results["int8_error"] = str(e)[:200]

    torch.save(model.state_dict(), os.path.join(HERE, "model.pt"))
    with open(os.path.join(HERE, "labels.json"), "w") as f:
        json.dump(CLASSES, f)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
