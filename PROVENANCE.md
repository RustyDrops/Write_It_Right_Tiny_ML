# Write It Right — Baseline Provenance
Built: 2026-11-17/18 (America/Chicago) · Status: local baseline only — NOT published or uploaded anywhere

## Dataset
- **EMNIST, `byclass` split** (62 classes: 0-9, A-Z, a-z), via torchvision 0.22.1 (`torchvision.datasets.EMNIST`), which downloads the official EMNIST gzip archive (562 MB, all splits).
- Source: EMNIST — Cohen, Afshar, Tapson & van Schaik (2017), "EMNIST: an extension of MNIST to handwritten letters," arXiv:1702.05373. Derived from **NIST Special Database 19** (https://www.nist.gov/srd/nist-special-database-19).
- License/source status: NIST Special Database 19 is a US-government (NIST) work distributed as freely usable / public domain; NIST's EMNIST page states no license terms and asks users to cite the paper (citation given above). Secondary registries list the EMNIST release as CC0/CC-BY-4.0-equivalent. Recorded conservatively as: **public-domain US government source data, attribution to Cohen et al. (2017) and NIST SD19 preserved here.**
- Splits used: official train split, stratified subsample capped at 1,500 samples/class = **93,000 train samples** (cap chosen for the ~20-min CPU training budget); official test split used **in full: 116,323 samples**.
- Preprocessing: EMNIST images are stored transposed relative to upright; each image was transposed to upright orientation, scaled to 0-1. Training-only augmentation: rotation ±8° (p=0.7), translation ±2 px (p=0.7). No other data.

## Terms compliance
- Training data, labels, and filtering came **only** from EMNIST as described above.
- **No** training data, labels, scores, rankings, or filtering from Claude/Muse or OpenAI/Luna outputs were used at any stage. Frontier models were not used to generate or select data.
- Nothing was published or uploaded anywhere.

## Model
- Architecture: small CNN, 101,118 parameters — Conv(1→32)+BN+Pool, Conv(32→64)+BN+Pool, Conv(64→128)+BN, global average pool, Linear(128→62). (`train2.py`)
- Training: PyTorch 2.7.1 CPU on this VM, AdamW (lr 1.5e-3, cosine), cross-entropy, batch 256, 4 epochs, seed 42. Wall time ~28 min under shared-CPU contention with parallel jobs (a first run, `train.py`, was lost to a silent process kill during epoch 4 — likely shared-memory pressure; v2 removed DataLoader workers, held data in memory, and checkpointed per epoch).
- Run history (train accuracy, augmented): v1 (OneCycle, heavier augmentation): ep1 29.6%, ep2 65.4%, ep3 69.9%, killed in ep4. v2: ep1 50.2%, ep2 68.3%, ep3 71.1%, ep4 72.1%.

## Results — full official EMNIST byclass test split (116,323 samples)
- **Overall accuracy: 76.5%**
- Letters only (52 classes): 70.0% · Lowercase only (26 classes): 77.0%
- Focus letters (lowercase):
  - **b: 89.8%** (n=853) — confused with '6' 3.9%, 'B' 2.2%; **b→d confusion <1%**
  - **d: 94.9%** (n=1,683) — no single confusion ≥1%; **d→b confusion <1%**
  - **p: 72.8%** (n=368) — confused with uppercase 'P' 20.9% (visually near-identical case pair), 'D' 1.1%; p→q <1%
  - **q: 40.4%** (n=505) — confused with '9' 35.6%, 'g' 7.3%, 'a' 4.8%, '8' 3.2%, 'Q' 3.0% — the model's clear weak spot (q/9/g are near-identical in EMNIST handwriting)
- Interpretation: the b/d reversal pair the product cares about is **not** a model problem (<1% cross-confusion). Weakness is concentrated in q (vs 9/g) and case-ambiguous pairs (p/P).

## Artifacts (in this folder)
- `write_it_right.onnx` — fp32 ONNX, **404,790 bytes (~0.39 MB)** — verified via ONNX Runtime on the full test split: **76.52%** (matches PyTorch 76.5%)
- `write_it_right_int8.onnx` — dynamic int8-quantized ONNX, **109,191 bytes (~0.10 MB)** — verified paired on a fixed 10,000-sample test subset (`onnx_eval_subset.json`): **int8 75.92% vs fp32 76.85%** on the same samples, i.e. quantization costs ~0.9 points at ¼ the size. (A full-split int8 eval was attempted twice and killed by shared-resource pressure on the VM both times; the paired subset check is the recorded evidence.)
- `model_v2.pt` — PyTorch checkpoint (405 KB) · `labels.json` — 62-class label order · `results.json` — full metrics · `train2.py`, `eval_onnx.py` — training/verification code

## Known limits / next levers
- 76.5% is a baseline, not a ceiling: training used only ~13% of the official train data (1,500/class cap) and 4 epochs. Full-data training is the obvious next lever, plus targeted work on q/9/g and case pairs.
- EMNIST is adult handwriting, not 5-8-year-old children's handwriting — a children's dataset audit is still required before this can grade real kids' writing.

---

# v2 — Full-data training + letters word-mode (2026-11-18)
Status: local iteration only — NOT published or uploaded anywhere. Supersedes v1 as the current model; v1 artifacts preserved.

## What changed and why
- **Training data: the FULL official EMNIST byclass train split — 697,932 samples** (no subsample cap), 10 epochs. Same dataset, source, and license status as v1 (NIST SD19, public-domain US government work; Cohen et al. 2017 citation above). No other data.
- **Key product insight:** in the real product (kids writing spelling words) the answer is always a letter — digits are never valid. v2 is therefore also evaluated in **word-mode**: logits masked to the 52 letter classes, case merged (P/p both correct for the letter p). This is the product-relevant metric.
- Architecture: same small-CNN family, modestly grown — added one Conv(128→128)+BN block at 7×7; **248,958 params** (v1: 101,118). Training: PyTorch CPU, AdamW (lr 2e-3, cosine over 10 epochs), batch 512, cross-entropy, seed 42. Augmentation: batched random affine (rotation ±8°, shift ±2 px, scale 0.95–1.05) — **no flips** (a flip would corrupt b/d). (`train_full.py`)
- Run resilience: the shared VM restarted repeatedly during the run (roughly every ~2 h), killing the process each time. Training was made resumable — deterministic per-epoch shuffle, full optimizer/scheduler checkpoint every 200 steps — and was resumed from checkpoint after each restart. Cumulative compute across all chunks ≈ 11 h. Per-epoch test-10k raw/word-mode progressed 83.2/91.1 (ep1) → 85.3/94.9 (ep8) → 85.3/94.8 (ep10).

## Terms compliance (v2)
- Training data, labels, and filtering came **only** from EMNIST as described above.
- **No** training data, labels, scores, rankings, or filtering from Claude/Muse or OpenAI/Luna outputs were used at any stage.
- Nothing was published or uploaded anywhere.

## Results — full official EMNIST byclass test split (116,323 samples), PyTorch
- **Raw 62-class accuracy: 85.5%** (v1: 76.5%, +9.0 pts)
- **Word-mode letters accuracy: 95.2%** (58,405 letter samples)
  - Word-mode by letter (case-merged, both cases as truth): **b 96.5% · d 98.3% · p 98.3% · q 90.8%**
  - Word-mode, lowercase truth only: b 95.9% · d 98.9% · p 95.9% · q 84.8%
- Raw-mode focus (lowercase truth, exact class — for v1 comparison): b 88.7%, d 98.5%, p 14.1%, q 45.1%
  - p's raw collapse is case ambiguity, not letter error: 81.3% of true lowercase p is predicted as uppercase 'P' (near-identical glyphs). In word-mode (case merged) p is 98.3%.
  - q's raw errors go mainly to '9' (40.2%) — a digit, masked out in word-mode, which lifts q to 90.8% (case-merged).
  - b/d reversal cross-confusion remains negligible, consistent with v1.

## Artifacts (v2, in this folder)
- `write_it_right_v2.onnx` — fp32 ONNX, **995,532 bytes (~0.95 MB)**
- `write_it_right_v2_int8.onnx` — dynamic int8 ONNX, **259,143 bytes (~0.25 MB)** — well under the 2 MB ceiling
- **Int8 verification** (paired, first 10,000 official test samples, `onnx_eval_v2_subset.json` via `eval_v2_int8.py`): int8 raw 85.40% / word-mode 94.91% vs fp32 raw 85.33% / word-mode 94.77% on the same samples — quantization cost ≈ 0 (int8 a hair higher on this subset, within noise). Int8 word-mode focus on the subset: b 97.0% · d 98.5% · p 98.7% · q 86.0% (smaller n per letter than the full split)
- `model_full_ep{1..10}.pt` — per-epoch weights · `model_full_last.pt` — final resume checkpoint · `results_v2.json` — full metrics · `train_full.py` — training/eval code

## Known limits (v2)
- Raw accuracy is plateauing (~85.5%): the remaining raw errors are dominated by genuinely ambiguous EMNIST glyphs (case pairs like p/P, o/O, and q/9/g shapes). Word-mode removes most of that ambiguity by design; further raw gains would need a materially larger model, breaking the micro brief.
- Weakest product letter remains **q** (word-mode 90.8% case-merged; 84.8% lowercase-truth). If a future iteration targets one letter, it's q.
- Still adult handwriting (EMNIST), not children's — the children's-dataset audit from v1 remains required before grading real kids' writing.
