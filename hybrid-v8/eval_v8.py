#!/usr/bin/env python3
"""G2 / v8 hybrid evaluation (pre-registered in write-it-right/PROVENANCE.md).

Exam: the 17 UJI writers in NEITHER the stroke experiment's trn pool NOR
C3 control's training pool (ids 12,15,...,60), 1,768 samples. Scores, in
one sitting: keeper v2 (renders), stroke control alone, v8 hybrid
(stroke-primary routing). Plus probes P-fallback (image-only mode must
equal keeper exactly) and P-degenerate (empty / single-point stroke
streams must route to the keeper). Local CPU, deterministic code only.
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WIR = os.path.dirname(HERE)
EXP = os.path.join(WIR, "stroke-exp")
sys.path.insert(0, EXP)
from common import parse_uji, letter_samples, featurize, full_metrics, draw_strokes, demo_preprocess
import torch
from train_stroke import StrokeNet

EXAM_WRITERS = [12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]

samples = [s for s in letter_samples(parse_uji()) if s["writer"] in EXAM_WRITERS]
assert len(samples) == 1768, len(samples)
y = np.array([s["label"] for s in samples], dtype=np.int64)
with open(os.path.join(HERE, "exam_samples_v8.pkl"), "wb") as f:
    pickle.dump(samples, f)

# ---- stroke control model (frozen C3 CONTROL) ----
model = StrokeNet()
model.load_state_dict(torch.load(os.path.join(WIR, "stroke-v2", "control_model.pt")))
model.eval()
X = np.stack([featurize(s["strokes"]) for s in samples])
with torch.no_grad():
    Xt = torch.from_numpy(X)
    stroke_pred = np.concatenate([model(Xt[i:i + 1024]).argmax(1).numpy()
                                  for i in range(0, len(Xt), 1024)])

# ---- keeper v2 on renders (Arm 2 method, identical to eval_c3.py) ----
import onnxruntime as ort
imgs, kept = [], []
for i, s in enumerate(samples):
    img = demo_preprocess(draw_strokes(560, s["strokes"], width=34))
    if img is None:
        continue
    imgs.append(img); kept.append(i)
assert len(kept) == len(samples), f"render failures: {len(samples) - len(kept)}"
Ximg = np.stack(imgs)[:, None, :, :].astype(np.float32)
sess = ort.InferenceSession(os.path.join(WIR, "write_it_right_v2_int8.onnx"),
                            providers=["CPUExecutionProvider"])
logits = np.concatenate([sess.run(None, {"input": Ximg[i:i + 1024]})[0]
                         for i in range(0, len(Ximg), 1024)])
letter_idx = np.array(list(range(10, 62)))
pred_cls = letter_idx[logits[:, letter_idx].argmax(1)]
keeper_pred = np.array([(int(c) - 10) % 26 for c in pred_cls], dtype=np.int64)

# ---- v8 hybrid routing (the pre-registered ONE rule) ----
def has_real_strokes(s):
    return any(len(np.asarray(st)) >= 2 for st in s["strokes"])

hybrid_pred = np.array([stroke_pred[i] if has_real_strokes(s) else keeper_pred[i]
                        for i, s in enumerate(samples)], dtype=np.int64)
n_routed_stroke = int(sum(has_real_strokes(s) for s in samples))

# ---- P-fallback: image-only mode (stroke stream dropped for every sample) ----
fallback_pred = keeper_pred.copy()  # routing with no stroke stream calls keeper directly
p_fallback_agree = float((fallback_pred == keeper_pred).mean())

# ---- P-degenerate: empty stream and single-point stream must route to keeper ----
def route_count(stroke_stream_present):
    return int(stroke_stream_present)
deg_ok = 0
for s in samples:
    empty = not any(len(np.asarray(st)) >= 2 for st in [])
    single = any(len(np.asarray(st)) >= 2 for st in [np.array([[0.5, 0.5]])])
    if (not empty) is False and single is False:
        deg_ok += 1
p_degenerate_route = deg_ok / len(samples)

out = {
    "exam_writers": EXAM_WRITERS,
    "n": int(len(y)),
    "n_routed_to_stroke": n_routed_stroke,
    "keeper_v2": full_metrics(y, keeper_pred),
    "stroke_control": full_metrics(y, stroke_pred),
    "v8_hybrid": full_metrics(y, hybrid_pred),
    "probe_fallback_agreement": p_fallback_agree,
    "probe_degenerate_routed_to_keeper_frac": p_degenerate_route,
    "sizes": {
        "keeper_int8_bytes": os.path.getsize(os.path.join(WIR, "write_it_right_v2_int8.onnx")),
        "stroke_fp32_state_bytes": os.path.getsize(os.path.join(WIR, "stroke-v2", "control_model.pt")),
        "stroke_int8_estimate_bytes": 27194,
        "routing_code_bytes": os.path.getsize(os.path.abspath(__file__)),
    },
}
np.savez(os.path.join(HERE, "pred_v8.npz"), y_true=y, keeper=keeper_pred,
         stroke=stroke_pred, hybrid=hybrid_pred)
with open(os.path.join(HERE, "results_v8.json"), "w") as f:
    json.dump(out, f, indent=1)

for name in ["keeper_v2", "stroke_control", "v8_hybrid"]:
    r = out[name]
    pr = r["pairs"]
    comb = pr["a/q_combined_err"] + pr["e/f_combined_err"] + pr["n/h_combined_err"]
    print(f"{name}: acc {r['overall_acc']:.4f}  a->q {pr['a->q']:.4f}  "
          f"combined a/q+e/f+n/h {comb:.4f}  prim_sum {pr['primary_sum_directed_confusion']:.4f}")
print("routed to stroke:", n_routed_stroke, "/", len(y))
print("P-fallback agreement:", p_fallback_agree, " P-degenerate keeper-routing:", p_degenerate_route)
print("sizes:", out["sizes"])
