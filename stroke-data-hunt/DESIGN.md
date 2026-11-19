# Stroke-data-hunt — Phase 2 pre-registration (2026-11-19)

Written BEFORE any combined-data training. Survey and license records:
`SOURCES.md` in this folder.

## Baseline to beat — R2-AUG-N12 (current best all-gates candidate)

From `stroke-v2/DESIGN.md` (Round 2 result), on the frozen C3 exam
(UJI writers id % 3 == 0, 2,080 samples, `stroke-v2/data_n12.npz`
X_test/y_test):

- overall **0.8933** (1,858 / 2,080), val best 0.9295
- **a→q = 0.0000**, primary directed sum **0.0750** (at the strict limit)
- worst letter h 0.7500; int8 27,194 B
- Recipe: 3,536 UJI train originals + 3,536 augmented copies
  (frozen Lever-2 augmentation, RNG `default_rng(20261120)`),
  featurized at N = 12, `train_stroke.py` unmodified, seed 0,
  checkpoint selected on the C3 validation split (624 samples).

## New training data (Phase 1 haul — licenses in SOURCES.md)

| Source | Samples | Writers | Notes |
|---|---|---|---|
| UJI train (C3 partition, unchanged) | 3,536 | 40 | as in R2 |
| Character Trajectories (UCI 175, CC BY 4.0) | 2,858 | **1** | velocity→position integration, y-flip; cursive single-stroke allographs; 20 letters (no f/i/j/k/t/x) |
| HWRT letters (Zenodo 50022, ODbL 1.0) | 3,638 | **192** | crowdsourced Detexify/write-math recordings; all 26 letters (lowercase 't' absent — its 56 samples are uppercase T, case-merged) |
| **Combined originals** | **10,032** | 40 + 1 + 192 | 2.84× the R2 originals |

Provenance of both new corpora is disjoint from UJI (different
devices, studies, writers), so no UJI exam-writer data can be present;
no near-duplicate dedupe against the exam is applicable beyond this
provenance check. Validation (624) and exam (2,080) arrays are copied
**unchanged** from `data_n12.npz` and asserted array-identical.

## Candidate 1 — H1-COMBINED-AUG-N12

- Data: originals = UJI train ++ Character Trajectories ++ HWRT
  (in that order), featurized at N = 12 with `common.featurize`;
  plus ONE augmented copy of each original, produced by the frozen
  Lever-2 augmentation code (identical transforms and draw order as
  `stroke-v2/rounds/prep_aug.py`), RNG `np.random.default_rng(20261120)`
  consumed in combined-sample order. Because UJI samples come first,
  their augmented copies must reproduce R2's augmented block exactly —
  this is a **pipeline sanity gate** (asserted in prep; failure aborts
  before training). X_train = originals ++ augmented = 20,064 rows.
- Training: `stroke-exp/train_stroke.py` **unmodified** (seed 0,
  Adam 1e-3, batch 64, ≤ 60 epochs, early stop patience 10 on the
  same C3 validation split, best checkpoint restored). The trainer's
  built-in final pass is the candidate's ONE exam evaluation.
- **Gates (all must hold):**
  1. overall exam accuracy **> 0.8933** (≥ 1,859 / 2,080)
  2. **a→q == 0.0** exactly
  3. primary directed sum (a→q + q→a + e→f + f→e + n→h + h→n) **≤ 0.0750**
  4. int8 size ≤ 0.5 MB (27,194 B expected — architecture unchanged)
- Also reported: worst-letter accuracy, per-letter table, val best.

## Follow-up lever (only if Candidate 1 clears all gates)

ONE follow-up, registered in an addendum to this file **before** it
runs, chosen from: (a) plain (unaugmented) combined data at N = 12;
(b) a second augmented copy per original (aug ×2); (c) N = 16 on the
combined augmented data. Same gates, one exam evaluation.

## Discipline

Local CPU only, 0 Colab units. No exam tuning, no seed re-rolls.
Keepers, the v8 hybrid, the demo, ADOPTION.md, and all stroke-v2
files are untouched. Results appended below after the runs.

## Results

### Candidate 1 — H1-COMBINED-AUG-N12: PASS, all gates

Trained exactly as pre-registered (early stop epoch 38, val best
0.9487), one exam evaluation. (Process note: the first launch was
killed by a service restart at ~epoch 10, before any exam evaluation
existed; the identical run was relaunched detached and completed.
Logged as a deviation of execution, not of protocol.)

| Gate | R2-AUG-N12 (baseline) | H1-COMBINED-AUG-N12 |
|---|---|---|
| Overall (> 0.8933) | 0.8933 (1,858) | **0.9125** (1,898) ✓ |
| a→q (== 0.0) | 0.0000 | **0.0000** ✓ |
| Primary directed sum (≤ 0.0750) | 0.0750 | **0.0500** ✓ |
| int8 (≤ 0.5 MB) | 27,194 B | 27,194 B ✓ |
| Worst letter | h 0.7500 | k 0.7875 |

Directed pairs: a→q 0, q→a 0, e→f 0.0125, f→e 0, n→h 0.0375, h→n 0.
Per-letter floor rose across the board (no letter below 0.7875;
c = 1.000, o = 0.975). +1.92 pts over the baseline from data alone —
the largest single gain of any lever tried on this exam, and the
first time the val→exam gap was attacked by adding *writers*
(192 HWRT writers + CT's one very different cursive writer) rather
than by transforming the same 40 writers' letters.

### Follow-up addendum — H2-COMBINED-AUG-N16 (registered BEFORE this run)

Candidate 1 cleared all gates, so per the pre-registration one
follow-up lever runs. Choice: **option (c), N = 16** on the identical
combined augmented data. Rationale: at UJI-only scale N=16 twice
posted the best overall of the plain candidates (0.8808 on the C3
split, 88.22% on the old split) but was pair-gate-fragile; Candidate
1's primary sum (0.0500) now has margin, and with 2.84× the originals
the richer featurization is the untested axis. Everything else is
identical to Candidate 1: same combined originals in the same order,
same frozen augmentation with the same RNG stream (augmented strokes
are N-independent; only featurization changes), same val/exam arrays,
unmodified `train_stroke.py`, seed 0, one exam evaluation.
Sanity gate for this prep: the UJI originals block featurized at
N=16 must equal `stroke-v2/data_n16.npz` X_train exactly.
Gates: the same four (overall > 0.8933 — and, to count as an
improvement over Candidate 1, > 0.9125; a→q == 0.0; sum ≤ 0.0750;
int8 ≤ 0.5 MB).

**Deviation (logged 2026-11-19, before the corrected run):** the
first H2 attempt was invalid and is NOT an evaluation of the
registered candidate. Its prep script copied X_val/X_test from
`data_n12.npz` (featurized at N=12) while X_train was featurized at
N=16 — a train/eval featurization mismatch (the length-agnostic conv
net ran anyway; val best 0.8718, "TEST acc" 0.7721). The registered
candidate featurizes the exam at its own N, as every prior candidate
did, so that run measured a different, broken system. No design
choice was tuned on it. Its artifacts are preserved as
`h2bug_*` / `train_h2bug.log`. Fix: val/test arrays taken from
`stroke-v2/data_n16.npz` (same split; y_val/y_test asserted identical
to `data_n12.npz`). The corrected H2 run below is the candidate's
one evaluation.

### Follow-up result — H2-COMBINED-AUG-N16: passes baseline gates, NOT an improvement

Corrected run exactly as registered (early stop epoch 24, val best
0.9487), one exam evaluation:

| Gate | R2 baseline | H1 (N=12) | H2 (N=16) |
|---|---|---|---|
| Overall | 0.8933 | **0.9125** | 0.9062 ✓ vs baseline, ✗ vs H1 |
| a→q | 0.0000 | 0.0000 | 0.0000 ✓ |
| Primary sum | 0.0750 | **0.0500** | 0.0750 ✓ (at limit) |
| int8 | 27,194 B | 27,194 B | 27,194 B ✓ |
| Worst letter | h 0.7500 | k 0.7875 | k 0.8125 |

Directed pairs (H2): q→a 0.0125, f→e 0.0500, n→h 0.0125, rest 0.
Reading: at combined-data scale, N=16 again trades pair cleanliness
for nothing — overall lower than N=12 AND the pair sum back at the
limit, though its per-letter floor is the highest yet (k 0.8125).
**H1-COMBINED-AUG-N12 is the best candidate of this hunt** and the
one that would be proposed (adoption is, as always, a separate
decision; nothing here modifies the hybrid or ADOPTION.md).

## Distance-to-95 verdict

- Best now: **0.9125** on the frozen C3 exam. The 95% target needs
  +3.75 pts more (78 more exam letters net). The hunt bought
  +1.92 pts — the biggest single gain on this exam to date — by
  *adding writers*, exactly as the plateau analysis predicted.
- The val→exam gap did NOT close: H1 val best 0.9487 vs exam
  0.9125 (3.6 pts), same as R2's gap. 233 training writers (40 UJI
  + 192 HWRT + 1 CT) still leave unseen-writer generalization as
  the binding constraint — but the level shifted up ~2 pts at both
  ends, which says the constraint is now *quantity/quality of
  writer-diverse data*, not a hard wall at 89%.
- Remaining per-letter weakness is concentrated: k (0.79–0.81 in
  both candidates), then i/r/y/n/p in the 0.83–0.88 band. k is a
  multi-stroke letter that neither new corpus covers well (CT has
  no k at all; HWRT k = 154 crowdsourced samples of mixed quality).
- What would plausibly close the rest: (1) UNIPEN 1b+1c
  (~54k isolated letters, hundreds of writers) — blocked only by
  its encumbered terms (SOURCES.md records that §II.b permits
  training commercial recognizers under obligations; obtaining it
  from iUF under those terms is a human decision, not an
  engineering one); (2) more writers of *printed* letters at HWRT
  scale or beyond — no other permissively-licensed corpus was
  found; (3) the family's captured samples for the product's real
  distribution (few hundred samples won't move this exam, but they
  target the errors the family actually hits).
- Honest bottom line: 95% on this exam is not reachable with the
  permissively-licensed public data that exists today. 91.25% is
  what the license-clean web had. The gap to 95% is now a
  licensing/procurement question (UNIPEN) plus a data-collection
  question (family captures), not a modeling question.

Artifacts in this folder: SOURCES.md, DESIGN.md (this file),
convert_char_traj.py, convert_hwrt.py, prep_combined.py,
prep_combined_n16.py, data_hunt_n12.npz, data_hunt_n16.npz,
converted/{char_traj_samples.pkl, hwrt_letter_samples.pkl},
h1_combined_aug_n12_{model.pt,pred.npz,info.json}, h1_metrics.json,
h2_combined_aug_n16_{model.pt,pred.npz,info.json}, h2_metrics.json,
h2bug_* (invalid-run artifacts, preserved), train_h1.log,
train_h2.log, train_h2bug.log, raw/ (UCI zip + HWRT tar kept as
provenance; the 1.3 GB extracted HWRT CSVs were deleted after
conversion — re-extractable from raw/hwrt.tar).
