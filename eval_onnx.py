#!/usr/bin/env python3
"""Verify exported ONNX (fp32 + int8) accuracy on full official EMNIST byclass test."""
import json, os
import numpy as np
import onnxruntime as ort
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
ds = datasets.EMNIST(os.path.join(HERE, "data"), split="byclass", train=False, download=True)
X = ds.data.numpy(); y = ds.targets.numpy()
# upright transpose + scale, same as training
X = np.transpose(X, (0, 2, 1)).astype(np.float32) / 255.0
X = X[:, None, :, :]
out = {}
for name in ["write_it_right.onnx", "write_it_right_int8.onnx"]:
    sess = ort.InferenceSession(os.path.join(HERE, name), providers=["CPUExecutionProvider"])
    preds = []
    for i in range(0, len(y), 1024):
        logits = sess.run(None, {"input": X[i:i+1024]})[0]
        preds.append(logits.argmax(1))
    pred = np.concatenate(preds)
    acc = float((pred == y).mean())
    out[name] = acc
    print(name, f"{acc:.4f}", flush=True)
with open(os.path.join(HERE, "onnx_eval.json"), "w") as f:
    json.dump(out, f, indent=1)
print("DONE")
