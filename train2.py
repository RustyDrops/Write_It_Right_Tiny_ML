#!/usr/bin/env python3
"""Write It Right v2 (hardened): in-memory EMNIST byclass, single process,
checkpoint per epoch, cosine LR, lighter augmentation."""
import json, time, random, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
torch.set_num_threads(4)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
CAP = 1500; EPOCHS = 4; BATCH = 256; LR = 1.5e-3
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]

def load_arrays(train=True):
    ds = datasets.EMNIST(DATA, split="byclass", train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy()  # images stored transposed

class Arr(Dataset):
    def __init__(self, X, y, train=False):
        self.X = X; self.y = y; self.train = train
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        img = torch.from_numpy(self.X[i].T.copy()).float().unsqueeze(0) / 255.0  # transpose to upright
        if self.train:
            if random.random() < 0.7:
                img = transforms.functional.rotate(img, random.uniform(-8, 8))
            if random.random() < 0.7:
                img = transforms.functional.affine(img, 0, [random.randint(-2, 2), random.randint(-2, 2)], 1.0, 0)
        return img, int(self.y[i])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, 62)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

def evaluate(model, loader):
    model.eval(); conf = np.zeros((62, 62), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1).numpy()
            for t, p in zip(y.numpy(), pred): conf[t, p] += 1
    return conf

def main():
    t0 = time.time()
    Xtr, ytr = load_arrays(True); Xte, yte = load_arrays(False)
    idx = []
    for c in range(62):
        ci = np.where(ytr == c)[0]
        np.random.RandomState(SEED + c).shuffle(ci)
        idx.extend(ci[:CAP].tolist())
    idx = np.array(idx); np.random.shuffle(idx)
    train_ds = Arr(Xtr[idx], ytr[idx], train=True)
    test_ds = Arr(Xte, yte, train=False)
    print(f"train {len(train_ds)} test {len(test_ds)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, num_workers=0)
    model = Net(); print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))
    lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train(); tot = cor = 0
        for x, y in train_loader:
            opt.zero_grad(); out = model(x); loss = lossf(out, y)
            loss.backward(); opt.step(); sched.step()
            tot += len(y); cor += (out.argmax(1) == y).sum().item()
        torch.save(model.state_dict(), os.path.join(HERE, "model_v2.pt"))
        print(f"epoch {ep+1}/{EPOCHS} train_acc={cor/tot:.4f} elapsed={time.time()-t0:.0f}s ckpt saved", flush=True)
    conf = evaluate(model, test_loader)
    acc = np.trace(conf) / conf.sum()
    li = list(range(10, 62)); loi = list(range(36, 62))
    res = {"overall_test_acc": float(acc),
           "letters_acc": float(conf[np.ix_(li, li)].trace() / conf[li].sum()),
           "lowercase_acc": float(conf[np.ix_(loi, loi)].trace() / conf[loi].sum()),
           "params": sum(p.numel() for p in model.parameters()),
           "train_samples": len(train_ds), "test_samples": int(conf.sum()),
           "epochs": EPOCHS, "focus_bdpq": {}, "train_seconds": time.time() - t0}
    for ch in "bdpq":
        ci = CLASSES.index(ch); row = conf[ci]; tot_ = row.sum()
        res["focus_bdpq"][ch] = {"n": int(tot_), "acc": float(row[ci] / tot_),
            "confused_with": {CLASSES[j]: float(row[j] / tot_) for j in range(62) if j != ci and row[j] / tot_ >= 0.01}}
    print(json.dumps(res, indent=1), flush=True)
    model.eval()
    onnx_path = os.path.join(HERE, "write_it_right.onnx")
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), onnx_path, input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}}, opset_version=13)
    res["onnx_fp32_bytes"] = os.path.getsize(onnx_path)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qp = os.path.join(HERE, "write_it_right_int8.onnx")
        quantize_dynamic(onnx_path, qp, weight_type=QuantType.QInt8)
        res["onnx_int8_bytes"] = os.path.getsize(qp)
    except Exception as e:
        res["int8_error"] = str(e)[:200]
    with open(os.path.join(HERE, "labels.json"), "w") as f: json.dump(CLASSES, f)
    with open(os.path.join(HERE, "results.json"), "w") as f: json.dump(res, f, indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
