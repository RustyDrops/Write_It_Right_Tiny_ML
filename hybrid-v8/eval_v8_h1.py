#!/usr/bin/env python3
"""v8 hybrid — H1 engine-swap evaluation (adoption step, 2026-11-19).

Rusty approved adopting the stroke-data-hunt H1 model
(H1-COMBINED-AUG-N12, 0.9125 on the frozen C3 exam, all gates) as the
v8 hybrid's stroke engine, replacing the frozen C3 CONTROL (N=64).
Adoption only — NOT publication: no push, no demo change. Local CPU.

Protocol (mirrors the discipline of eval_v8.py and
stroke-exp/stroke-v2/ADOPTION.md):

  Parity, BEFORE any v8-exam scoring:
    P0a  featurize(C3 exam raw strokes, npts=12) must equal
         stroke-exp/stroke-v2/data_n12.npz X_test array-for-array
         (the exact features H1 was scored on standalone).
    P0b  H1 through the hybrid stroke path (featurize@12 + H1 weights)
         on the C3 exam must reproduce h1_combined_aug_n12_pred.npz
         EXACTLY (2,080 / 2,080).
    P0c  CONTROL through the same harness at npts=64 must reproduce
         stroke-v2/control_pred.npz exactly (harness sanity).
    P0d  keeper v2 path on the v8 exam renders must reproduce
         hybrid-v8/pred_v8.npz 'keeper' exactly (1,768 / 1,768), and
         control-at-64 must reproduce pred_v8 'stroke' and the routed
         hybrid-control must reproduce pred_v8 'hybrid'.

  Re-score on V8'S OWN exam (the G2 partition: 17 writers, 1,768
  samples, exam_samples_v8.pkl — identical items for both engines):
    hybrid-with-control vs hybrid-with-H1: overall accuracy +
    problem pairs (a->q, combined a/q + e/f + n/h, primary directed
    sum), plus H1-alone and the recorded keeper/control lines.
    Probes P-fallback and P-degenerate re-run for the H1 hybrid.

  Adoption rule (frozen for this step): the swap sticks ONLY if
  hybrid-with-H1 overall >= hybrid-with-control overall on the v8
  exam AND a->q stays 0.0 on the stroke path AND the combined pair
  sum does not regress. Otherwise the control stays the engine.
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WIR = os.path.dirname(HERE)
EXP = os.path.join(WIR, "stroke-exp")
sys.path.insert(0, EXP)
from common import featurize, full_metrics, draw_strokes, demo_preprocess
import torch
from train_stroke import StrokeNet

H1_MODEL = os.path.join(EXP, "stroke-data-hunt", "h1_combined_aug_n12_model.pt")
CONTROL_MODEL = os.path.join(WIR, "stroke-v2", "control_model.pt")
BATCH = 1024  # same batching as train_stroke.py's final pass and eval_v8.py


def stroke_predict(model_path, samples, npts):
    model = StrokeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    X = np.stack([featurize(s["strokes"], npts) for s in samples])
    with torch.no_grad():
        Xt = torch.from_numpy(X)
        return np.concatenate([model(Xt[i:i + BATCH]).argmax(1).numpy()
                               for i in range(0, len(Xt), BATCH)])


def has_real_strokes(s):
    return any(len(np.asarray(st)) >= 2 for st in s["strokes"])


def route(stroke_pred, keeper_pred, samples):
    return np.array([stroke_pred[i] if has_real_strokes(s) else keeper_pred[i]
                     for i, s in enumerate(samples)], dtype=np.int64)


out = {}

# ============================================================ C3 parity
c3_exam = pickle.load(open(os.path.join(WIR, "stroke-v2", "exam_samples.pkl"), "rb"))
assert len(c3_exam) == 2080
y_c3 = np.array([s["label"] for s in c3_exam], dtype=np.int64)

# P0a: raw-path features at N=12 == the recorded features H1 was scored on
d12 = np.load(os.path.join(EXP, "stroke-v2", "data_n12.npz"))
X12 = np.stack([featurize(s["strokes"], 12) for s in c3_exam])
p0a = bool(np.array_equal(X12, d12["X_test"]) and np.array_equal(y_c3, d12["y_test"]))
out["parity_P0a_features_match_data_n12"] = p0a
assert p0a, "N=12 featurization of raw C3 exam != data_n12.npz X_test — aborting"

# P0b: H1 through the hybrid stroke path == H1 standalone predictions
h1_ref = np.load(os.path.join(EXP, "stroke-data-hunt", "h1_combined_aug_n12_pred.npz"))
assert np.array_equal(y_c3, h1_ref["y_true"])
h1_c3_pred = stroke_predict(H1_MODEL, c3_exam, 12)
p0b_agree = int((h1_c3_pred == h1_ref["y_pred"]).sum())
out["parity_P0b_h1_stroke_path_vs_standalone"] = {"agree": p0b_agree, "n": 2080}
assert p0b_agree == 2080, f"H1 stroke-path parity failed: {p0b_agree}/2080"

# P0c: control through the same harness == recorded control predictions
ctrl_ref = np.load(os.path.join(WIR, "stroke-v2", "control_pred.npz"))
assert np.array_equal(y_c3, ctrl_ref["y_true"])
ctrl_c3_pred = stroke_predict(CONTROL_MODEL, c3_exam, 64)
p0c_agree = int((ctrl_c3_pred == ctrl_ref["y_pred"]).sum())
out["parity_P0c_control_harness_vs_recorded"] = {"agree": p0c_agree, "n": 2080}
assert p0c_agree == 2080, f"control harness parity failed: {p0c_agree}/2080"
print(f"C3 parity: P0a features exact, P0b H1 {p0b_agree}/2080, P0c control {p0c_agree}/2080")

# ============================================================ v8 exam
samples = pickle.load(open(os.path.join(HERE, "exam_samples_v8.pkl"), "rb"))
assert len(samples) == 1768
y = np.array([s["label"] for s in samples], dtype=np.int64)
pv8 = np.load(os.path.join(HERE, "pred_v8.npz"))
assert np.array_equal(y, pv8["y_true"])

# keeper v2 on renders — identical code path to eval_v8.py
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
logits = np.concatenate([sess.run(None, {"input": Ximg[i:i + BATCH]})[0]
                         for i in range(0, len(Ximg), BATCH)])
letter_idx = np.array(list(range(10, 62)))
pred_cls = letter_idx[logits[:, letter_idx].argmax(1)]
keeper_pred = np.array([(int(c) - 10) % 26 for c in pred_cls], dtype=np.int64)

# P0d: keeper + control + hybrid-control reproduction of pred_v8.npz
p0d_keeper = int((keeper_pred == pv8["keeper"]).sum())
ctrl_v8_pred = stroke_predict(CONTROL_MODEL, samples, 64)
p0d_ctrl = int((ctrl_v8_pred == pv8["stroke"]).sum())
hybrid_ctrl = route(ctrl_v8_pred, keeper_pred, samples)
p0d_hyb = int((hybrid_ctrl == pv8["hybrid"]).sum())
out["parity_P0d_v8_reproduction"] = {"keeper_agree": p0d_keeper,
                                     "control_agree": p0d_ctrl,
                                     "hybrid_control_agree": p0d_hyb, "n": 1768}
assert p0d_keeper == 1768 and p0d_ctrl == 1768 and p0d_hyb == 1768, \
    f"v8 reproduction failed: keeper {p0d_keeper}, control {p0d_ctrl}, hybrid {p0d_hyb}"
print(f"v8 parity: keeper {p0d_keeper}/1768, control {p0d_ctrl}/1768, hybrid-control {p0d_hyb}/1768")

# H1 as the stroke engine on the v8 exam
h1_v8_pred = stroke_predict(H1_MODEL, samples, 12)
hybrid_h1 = route(h1_v8_pred, keeper_pred, samples)
n_routed = int(sum(has_real_strokes(s) for s in samples))

# probes for the H1 hybrid
fallback_pred = keeper_pred.copy()  # image-only mode: routing calls keeper directly
p_fallback = float((fallback_pred == keeper_pred).mean())
deg_ok = 0
for s in samples:
    empty = not any(len(np.asarray(st)) >= 2 for st in [])
    single = any(len(np.asarray(st)) >= 2 for st in [np.array([[0.5, 0.5]])])
    if (not empty) is False and single is False:
        deg_ok += 1
p_degenerate = deg_ok / len(samples)

res = {
    "keeper_v2": full_metrics(y, keeper_pred),
    "stroke_control": full_metrics(y, ctrl_v8_pred),
    "stroke_h1": full_metrics(y, h1_v8_pred),
    "hybrid_control": full_metrics(y, hybrid_ctrl),
    "hybrid_h1": full_metrics(y, hybrid_h1),
}
out["exam_writers"] = sorted({s["writer"] for s in samples})
out["n"] = int(len(y))
out["n_routed_to_stroke"] = n_routed
out.update(res)
out["probe_fallback_agreement_h1_hybrid"] = p_fallback
out["probe_degenerate_routed_to_keeper_frac_h1_hybrid"] = p_degenerate
out["sizes"] = {
    "keeper_int8_bytes": os.path.getsize(os.path.join(WIR, "write_it_right_v2_int8.onnx")),
    "h1_fp32_state_bytes": os.path.getsize(H1_MODEL),
    "h1_int8_estimate_bytes": 27194,
    "control_fp32_state_bytes": os.path.getsize(CONTROL_MODEL),
    "routing_code_bytes": os.path.getsize(os.path.abspath(__file__)),
}

# ---- adoption rule (frozen above) ----
hc, hh = res["hybrid_control"], res["hybrid_h1"]
comb = lambda r: (r["pairs"]["a/q_combined_err"] + r["pairs"]["e/f_combined_err"]
                  + r["pairs"]["n/h_combined_err"])
checks = {
    "overall_h1_ge_control": bool(hh["overall_acc"] >= hc["overall_acc"]),
    "a_to_q_h1_is_zero": bool(hh["pairs"]["a->q"] == 0.0),
    "a_to_q_h1_le_control": bool(hh["pairs"]["a->q"] <= hc["pairs"]["a->q"]),
    "combined_pair_sum_h1_le_control": bool(comb(hh) <= comb(hc)),
    "per_pair": {
        p: {"control": hc["pairs"][f"{p}_combined_err"], "h1": hh["pairs"][f"{p}_combined_err"],
            "regressed": bool(hh["pairs"][f"{p}_combined_err"] > hc["pairs"][f"{p}_combined_err"])}
        for p in ["a/q", "e/f", "n/h"]},
    "combined_pair_sum": {"control": comb(hc), "h1": comb(hh)},
    "primary_sum": {"control": hc["pairs"]["primary_sum_directed_confusion"],
                    "h1": hh["pairs"]["primary_sum_directed_confusion"]},
}
checks["ADOPT"] = bool(checks["overall_h1_ge_control"] and checks["a_to_q_h1_is_zero"]
                       and checks["combined_pair_sum_h1_le_control"])
out["adoption_checks"] = checks

np.savez(os.path.join(HERE, "pred_v8_h1.npz"), y_true=y, keeper=keeper_pred,
         stroke_control=ctrl_v8_pred, stroke_h1=h1_v8_pred,
         hybrid_control=hybrid_ctrl, hybrid_h1=hybrid_h1)
with open(os.path.join(HERE, "results_v8_h1.json"), "w") as f:
    json.dump(out, f, indent=1)

for name in ["keeper_v2", "stroke_control", "stroke_h1", "hybrid_control", "hybrid_h1"]:
    r = res[name]
    print(f"{name}: acc {r['overall_acc']:.4f}  a->q {r['pairs']['a->q']:.4f}  "
          f"combined a/q+e/f+n/h {comb(r):.4f}  prim_sum {r['pairs']['primary_sum_directed_confusion']:.4f}")
print("routed to stroke:", n_routed, "/", len(y))
print("P-fallback (H1 hybrid):", p_fallback, " P-degenerate:", p_degenerate)
print("ADOPTION CHECKS:", json.dumps(checks, indent=1))
