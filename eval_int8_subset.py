#!/usr/bin/env python3
"""Paired int8 vs fp32 check on first 10,000 official test samples (deterministic subset)."""
import json, os
import numpy as np
import onnxruntime as ort
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
ds = datasets.EMNIST(os.path.join(HERE, "data"), split="byclass", train=False, download=True)
X = ds.data.numpy()[:10000]; y = ds.targets.numpy()[:10000]
X = np.transpose(X, (0, 2, 1)).astype(np.float32) / 255.0
X = X[:, None, :, :]
out = {"subset": "first 10000 official test samples"}
for name in ["write_it_right.onnx", "write_it_right_int8.onnx"]:
    sess = ort.InferenceSession(os.path.join(HERE, name), providers=["CPUExecutionProvider"])
    preds = []
    for i in range(0, len(y), 1024):
        preds.append(sess.run(None, {"input": X[i:i+1024]})[0].argmax(1))
    acc = float((np.concatenate(preds) == y).mean())
    out[name] = acc
    print(name, f"{acc:.4f}", flush=True)
with open(os.path.join(HERE, "onnx_eval_subset.json"), "w") as f:
    json.dump(out, f, indent=1)
print("DONE")
