#!/usr/bin/env python3
"""Paired v2 fp32 vs int8 ONNX check on first 10,000 official test samples:
raw 62-class accuracy AND word-mode (letters-masked, case-merged) accuracy."""
import json, os
import numpy as np
import onnxruntime as ort
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
ds = datasets.EMNIST(os.path.join(HERE, "data"), split="byclass", train=False, download=True)
X = ds.data.numpy()[:10000]; y = ds.targets.numpy()[:10000]
X = np.transpose(X, (0, 2, 1)).astype(np.float32) / 255.0
X = np.ascontiguousarray(X[:, None, :, :])
LETTER_IDX = np.array(list(range(10, 62)))
def letter_id(ci): return (ci - 10) % 26

out = {"subset": "first 10000 official test samples", "n": int(len(y))}
for name in ["write_it_right_v2.onnx", "write_it_right_v2_int8.onnx"]:
    sess = ort.InferenceSession(os.path.join(HERE, name), providers=["CPUExecutionProvider"])
    logits = np.concatenate([sess.run(None, {"input": X[i:i+1024]})[0] for i in range(0, len(y), 1024)])
    raw = float((logits.argmax(1) == y).mean())
    pred_cls = LETTER_IDX[logits[:, LETTER_IDX].argmax(1)]
    m = y >= 10
    wm = float((np.array([letter_id(int(p)) for p in pred_cls[m]]) ==
                np.array([letter_id(int(t)) for t in y[m]])).mean())
    focus = {}
    for ch, lid in [("b", 1), ("d", 3), ("p", 15), ("q", 16)]:
        mm = m & (np.array([letter_id(int(t)) for t in y]) == lid)
        focus[ch] = float((np.array([letter_id(int(p)) for p in pred_cls[mm]]) == lid).mean())
    out[name] = {"raw_acc": raw, "wordmode_letters_acc": wm, "wordmode_focus": focus,
                 "bytes": os.path.getsize(os.path.join(HERE, name))}
    print(name, json.dumps(out[name]), flush=True)
with open(os.path.join(HERE, "onnx_eval_v2_subset.json"), "w") as f:
    json.dump(out, f, indent=1)
print("DONE")
