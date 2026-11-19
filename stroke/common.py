#!/usr/bin/env python3
"""Shared code for the Write It Right stroke experiment.

Implements the frozen design in DESIGN.md: UJIpenchars2 parsing, stroke
featurization, demo-mirroring rendering + preprocessing for keeper v2, and
confusion metrics. No LLM-generated data anywhere: UJIpenchars2 (CC BY 4.0)
and EMNIST (NIST SD19) only.
"""
import os, re, pickle
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
WIR = os.path.dirname(HERE)
UJI_TXT = os.path.join(HERE, "data", "uji2_raw", "ujipenchars2.txt")
NPTS = 64

# ---------------------------------------------------------------- UJI parsing
_WORD_RE = re.compile(r"^WORD (.+) ((?:trn|tst)_(?:UJI|UPV)_W(\d+)-(\d+))\s*$")

def parse_uji(path=UJI_TXT):
    """Return list of dicts: char, writer_set ('trn'/'tst'), site, writer (int),
    rep (int), strokes (list of (m,2) float arrays, x-right / y-down, UPV
    coordinates already homogenized /1.52)."""
    samples = []
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _WORD_RE.match(line)
            if m:
                if cur is not None:
                    samples.append(cur)
                char, sess, wnum, rep = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                wset, site = sess.split("_")[0], sess.split("_")[1]
                cur = {"char": char, "writer_set": wset, "site": site,
                       "writer": wnum, "rep": rep, "strokes": []}
                continue
            if cur is None:
                continue
            ls = line.strip()
            if ls.startswith("POINTS"):
                parts = ls.split("#", 1)[1].split()
                vals = np.array([float(v) for v in parts], dtype=np.float64)
                pts = vals.reshape(-1, 2)
                if cur["site"] == "UPV":
                    pts = pts / 1.52
                cur["strokes"].append(pts)
    if cur is not None:
        samples.append(cur)
    return samples

def letter_samples(samples):
    """ASCII letters only, case-merged label 0..25."""
    out = []
    for s in samples:
        c = s["char"]
        if len(c) == 1 and c.isascii() and c.isalpha():
            s = dict(s)
            s["label"] = ord(c.lower()) - 97
            out.append(s)
    return out

# ---------------------------------------------------------------- featurizer
def dedupe(pts):
    if len(pts) <= 1:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    return pts[keep]

def normalize_strokes(strokes):
    allpts = np.concatenate([s for s in strokes if len(s) > 0], axis=0)
    mn = allpts.min(axis=0)
    mx = allpts.max(axis=0)
    scale = max(mx[0] - mn[0], mx[1] - mn[1])
    if scale <= 0:
        scale = 1.0
    return [(s - mn) / scale for s in strokes]

def resample_stroke(pts, n):
    """Uniform-by-arc-length resample of one stroke to n points."""
    if len(pts) == 1 or n == 1:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, cum[-1], n)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.stack([x, y], axis=1)

def resample_sequence(strokes, npts=NPTS):
    """Resample a multi-stroke sequence to npts points total.
    Returns (pos (npts,2), pen_lift (npts,))."""
    strokes = [dedupe(s) for s in strokes if len(s) > 0]
    if not strokes:
        return np.zeros((npts, 2)), np.zeros(npts)
    strokes = normalize_strokes(strokes)
    k = len(strokes)
    lengths = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).sum() if len(s) > 1 else 0.0
                        for s in strokes])
    # allocation: min 2 per stroke (min 1 if budget is tight), rest by length
    base = 2 if 2 * k <= npts else 1
    alloc = np.full(k, base, dtype=int)
    remaining = npts - alloc.sum()
    if remaining > 0:
        if lengths.sum() > 0:
            extra = lengths / lengths.sum() * remaining
        else:
            extra = np.full(k, remaining / k)
        extra_i = np.floor(extra).astype(int)
        alloc += extra_i
        rem = npts - alloc.sum()
        order = np.argsort(-(extra - extra_i))
        for i in order[:rem]:
            alloc[i] += 1
    pos_parts, lift_parts = [], []
    for i, s in enumerate(strokes):
        rs = resample_stroke(s, int(alloc[i]))
        pos_parts.append(rs)
        lift = np.zeros(len(rs))
        if i > 0:
            lift[0] = 1.0
        lift_parts.append(lift)
    pos = np.concatenate(pos_parts, axis=0)
    lift = np.concatenate(lift_parts)
    # safety: allocation arithmetic should already give exactly npts
    if len(pos) > npts:
        pos, lift = pos[:npts], lift[:npts]
    elif len(pos) < npts:
        pad = npts - len(pos)
        pos = np.concatenate([pos, np.repeat(pos[-1:], pad, axis=0)])
        lift = np.concatenate([lift, np.zeros(pad)])
    return pos, lift

def featurize(strokes, npts=NPTS):
    """(5, npts) float32: x, y, dx, dy, pen_lift."""
    pos, lift = resample_sequence(strokes, npts)
    d = np.zeros_like(pos)
    d[1:] = np.diff(pos, axis=0)
    feats = np.stack([pos[:, 0], pos[:, 1], d[:, 0], d[:, 1], lift], axis=0)
    return feats.astype(np.float32)

# ---------------------------------------------------------------- rendering
def draw_strokes(size, strokes, width):
    """Render strokes (raw coords, any scale) to a float ink image (size x size),
    ink 1.0 on 0.0, fitting the strokes' bbox into the canvas with a small
    margin, preserving aspect. Round caps via vertex circles."""
    allpts = np.concatenate([s for s in strokes if len(s) > 0], axis=0)
    mn = allpts.min(axis=0); mx = allpts.max(axis=0)
    span = max(mx[0] - mn[0], mx[1] - mn[1])
    if span <= 0:
        span = 1.0
    margin = width  # px margin around the ink
    scale = (size - 2 * margin) / span
    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    r = width / 2.0
    for s in strokes:
        if len(s) == 0:
            continue
        p = (s - mn) * scale + margin
        if len(p) > 1:
            dr.line([tuple(q) for q in p], fill=255, width=int(round(width)), joint="curve")
        for q in p:
            dr.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0

def _js_round(x):
    return int(np.floor(x + 0.5))

def demo_preprocess(ink560):
    """Port of the demo's preprocess() (demo/build_demo.py): ink float array
    (560x560, ink intensity 0..1) -> (28,28) float32 ink, or None if blank.
    Downscale via PIL bicubic instead of canvas drawImage smoothing
    (documented in DESIGN.md); all geometry identical."""
    ink = ink560
    mask = ink > 0.08
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    minx, maxx, miny, maxy = xs.min(), xs.max(), ys.min(), ys.max()
    bw = maxx - minx + 1; bh = maxy - miny + 1
    scale = 20.0 / max(bw, bh)
    sw = max(1, _js_round(bw * scale)); sh = max(1, _js_round(bh * scale))
    crop = ink[miny:maxy + 1, minx:maxx + 1]
    patch_img = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
    patch_img = patch_img.resize((sw, sh), Image.BICUBIC)
    patch = np.asarray(patch_img, dtype=np.float64) / 255.0
    tot = patch.sum()
    if tot <= 0:
        return None
    yy, xx = np.mgrid[0:sh, 0:sw]
    cx = (xx * patch).sum() / tot; cy = (yy * patch).sum() / tot
    ox = max(0, min(28 - sw, _js_round(13.5 - cx)))
    oy = max(0, min(28 - sh, _js_round(13.5 - cy)))
    out = np.zeros((28, 28), dtype=np.float32)
    out[oy:oy + sh, ox:ox + sw] = patch.astype(np.float32)
    return out

# ---------------------------------------------------------------- metrics
PAIRS = [("a", "q"), ("e", "f"), ("n", "h"), ("b", "d"), ("p", "q")]
PRIMARY = [("a", "q"), ("e", "f"), ("n", "h")]

def lid(ch):
    return ord(ch) - 97

def pair_report(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rep = {}
    for a, b in PAIRS:
        ia, ib = lid(a), lid(b)
        ma = y_true == ia; mb = y_true == ib
        rep[f"{a}->{b}"] = float((y_pred[ma] == ib).mean()) if ma.any() else None
        rep[f"{b}->{a}"] = float((y_pred[mb] == ia).mean()) if mb.any() else None
        both = ma | mb
        rep[f"{a}/{b}_combined_err"] = float((y_pred[both] != y_true[both]).mean()) if both.any() else None
    prim = 0.0
    for a, b in PRIMARY:
        prim += rep[f"{a}->{b}"] + rep[f"{b}->{a}"]
    rep["primary_sum_directed_confusion"] = float(prim)
    return rep

def full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    per_letter = {}
    for i in range(26):
        m = y_true == i
        per_letter[chr(97 + i)] = {"acc": float((y_pred[m] == i).mean()) if m.any() else None,
                                   "n": int(m.sum())}
    return {"overall_acc": float((y_pred == y_true).mean()), "n": int(len(y_true)),
            "pairs": pair_report(y_true, y_pred), "per_letter": per_letter}

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
