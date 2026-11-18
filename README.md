# Write It Right — Tiny ML

A tiny handwriting-letter recognizer (0.25 MB, runs offline in a browser or on
a microcontroller-class device) built to check the letters young writers mix
up most: **b, d, p, q**. Trained only on public-domain data — no AI-generated
labels, no frontier-model output in the training data. See PROVENANCE.md.

## v2 results (full official EMNIST byclass test, 116,323 samples)

| Metric | v1 | v2 |
|---|---|---|
| Raw 62-class accuracy | 76.5% | **85.5%** |
| Word-mode accuracy (letters only, case-merged) | — | **95.2%** |
| Word-mode b / d / p / q | 89.8 / 94.9 / 72.8 / 40.4% (raw) | **96.5 / 98.3 / 98.3 / 90.8%** |

Word-mode is the product metric: when a child writes a word, digits are not
candidate answers and letter case doesn't matter. v1's scary-looking p and q
scores were mostly mode artifacts (p predicted as capital P; q predicted as
the digit 9).

- Model: 248,958-param CNN, trained on the full EMNIST byclass train split
  (697,932 samples, 10 epochs)
- int8 ONNX: 259,143 bytes; quantization cost ≈ 0 (paired 10k check:
  word-mode 94.91% int8 vs 94.77% fp32)
- Data: EMNIST (NIST Special Database 19) — public domain

## Honest limits

EMNIST is **adult** handwriting. This model has not been validated on
children's handwriting; a kids'-handwriting evaluation is required before
using it to grade real children's work. q remains the weakest letter.

## Files

- `train_full.py` — the v2 training script (resumable checkpoints)
- `results_v2.json` / `results.json` — full metrics for v2 / v1
- `onnx_eval_v2_subset.json` — int8-vs-fp32 paired evaluation
- `PROVENANCE.md` — data sources, licenses, and method
- The int8 ONNX model itself is a binary and is not in this repo (the GitHub
  push path here is text-only); it is available on request.
