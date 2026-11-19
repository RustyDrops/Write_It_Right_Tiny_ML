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

---

# v3 — GPU self-improvement attempt on Colab (2026-11-18)
Status: local iteration only — NOT published or uploaded anywhere. **Verdict: NOT the keeper — v2 remains the current model** (see keeper rule below).

## What was tried (vs v2)
- **Compute:** Google Colab T4 GPU via the MCP route (driver `colab_v3_job.py`), EMNIST downloaded on the runtime itself from the official source via torchvision. Training took 764 s in-kernel (~13 min for 16 epochs vs ~11 h cumulative on the shared VM for v2). **Colab cost: 0.28 compute units** (balance 597.73 → 597.45; runtime deleted, verified via list_runtimes + `colab usage`, 0 active assignments).
- **Augmentation:** v2's affine widened (rotation ±10°, shift ±2.5 px, scale 0.92–1.08) **plus elastic distortion** (smooth 5×5-control-point displacement field, ≤~1.8 px, applied per-sample with p=0.5). No flips.
- **Targeted emphasis:** plain oversampling of real EMNIST samples — lowercase 'q' duplicated ×2 per epoch (+2,494 copies), lowercase 'g' (q's top letter rival) +50% (+1,594 copies). No threshold biasing, no synthetic data.
- **Architecture:** modestly larger CNN, channels 40/80/160/160, **386,062 params** (v2: 248,958). 16 epochs, AdamW (lr 2e-3, cosine), batch 1024, seed 42. (`train_v3.py`)
- **Discipline:** a stratified validation split (500/class = 31,000 samples) was carved from the TRAIN split only and used for checkpoint selection (best val word-mode: epoch 16, 93.81%). The official test split was evaluated **once**, on the selected checkpoint. Note the val split is class-stratified while the official test is ~50% digits, so val raw accuracy (~70%) is not comparable to test raw — val word-mode is the comparable tuning signal.

## Terms compliance (v3)
- Training data, labels, and filtering came **only** from EMNIST (NIST SD19, public-domain US government work; Cohen et al. 2017), same as v1/v2.
- **No** training data, labels, scores, rankings, or filtering from Claude/Muse or OpenAI/Luna outputs were used at any stage. Augmentation and oversampling are deterministic code operations on real EMNIST samples.
- Nothing was published or uploaded anywhere.

## Results — full official EMNIST byclass test split (116,323 samples), single evaluation
- **Raw 62-class accuracy: 84.0%** (v2: 85.5%, **−1.5 pts**)
- **Word-mode letters accuracy: 94.1%** (v2: 95.2%, **−1.1 pts**)
  - Word-mode by letter (case-merged): **b 97.9% (+1.4) · d 97.8% (−0.4) · p 98.1% (−0.2) · q 91.9% (+1.1)**
  - Word-mode, lowercase truth only: b 97.5% (+1.6) · d 98.5% (−0.4) · p 95.9% (±0.0) · q 86.7% (+2.0)
- Raw-mode focus (lowercase truth): b 87.6%, d 97.7%, p 4.1% (91.8% predicted as 'P' — case collapse deepened vs v2's 81.3%), q 47.7% (still 41.0% to '9', masked in word-mode).
- **Keeper rule (pre-registered): v3 replaces v2 only if word-mode improves ≥0.5 pt with no focus letter dropping >1 pt. FAILED on the headline metric (−1.1 pts), so v2 stays the keeper.** The targeted lever worked in isolation — q improved on both word-mode views and b improved — but the package as a whole (stronger augmentation and/or 31k fewer training samples due to the val carve) cost more general accuracy than the emphasis gained.

## Artifacts (v3, in this folder)
- `write_it_right_v3_int8.onnx` — dynamic int8 ONNX, **396,335 bytes (~0.39 MB)** — under the 2 MB ceiling
- `write_it_right_v3.onnx` — fp32 ONNX, 1,543,244 bytes · `model_v3_best.pt` — selected checkpoint (epoch 16) · `results_v3.json` — full metrics · `train_v3.py` — training code · `colab_v3_job.py` — Colab driver · `train_v3.log` — runtime training log
- **Export note:** the runtime's `torch.onnx.export` failed after training + the single test evaluation (the runtime's newer torch requires the `onnxscript` package, which was not installed). The printed test results were recovered from the runtime log and the checkpoint was downloaded intact; ONNX export was then done locally from those exact weights with `export_v3_local.py` (VM torch 2.7.1, same exporter as v2). The test split was **not** re-evaluated for the headline metrics.
- **Int8 verification** (paired, first 10,000 official test samples, `onnx_eval_v3_subset.json` via `eval_v3_int8.py`): int8 raw 84.26% / word-mode 94.09% vs fp32 raw 83.96% / word-mode 93.97% on the same samples — quantization cost ≈ 0 (int8 a hair higher, within noise; same pattern as v2). The subset fp32 figures also cross-check the runtime's full-test evaluation (84.0% / 94.1%).

## Known limits (v3)
- Same as v2: EMNIST is adult handwriting — the children's-handwriting dataset audit is still required before grading real kids' writing.
- If a v4 is attempted: keep the q/g oversampling (it demonstrably helped q and b), but revert to v2-strength augmentation and train on the full train split (select epochs by a fixed schedule or a smaller val carve) — the evidence points at the heavier elastic/affine augmentation as the likely cost.

---

# v4 — v2 recipe + v3's q/g oversampling (2026-11-18)
Status: local iteration only — NOT published or uploaded anywhere.

## Pre-registration (written BEFORE training)
- **Recipe (exactly two ingredients, no others):**
  1. v2's architecture (CNN 32/64/128/128, 248,958 params) and v2-strength affine augmentation (rotation ±8°, shift ±2 px, scale 0.95–1.05, applied to every sample; **no elastic distortion, no widened affine**).
  2. v3's confusable-pair oversampling, unchanged: lowercase 'q' duplicated ×2 per epoch, lowercase 'g' +50% per epoch. Plain oversampling of real EMNIST samples; no threshold biasing, no synthetic data.
- **Training:** full official EMNIST byclass train split (697,932 samples). A small stratified validation slice (80/class = 4,960 samples, seed 20261119) is carved from train and used ONLY for checkpoint selection (best val word-mode) — not v3's 31k carve. 12 epochs, AdamW (lr 2e-3, cosine), batch 1024, seed 42.
- **Evaluation discipline:** the official test split is evaluated **exactly once**, on the selected checkpoint (raw + word-mode + b/d/p/q + lowercase-truth focus), followed by the paired int8-vs-fp32 subset check on the first 10,000 test samples, as in v2/v3.
- **Keeper gates (ALL must pass, else v4 is rejected and v2 stays):**
  1. Word-mode letters accuracy ≥ 95.7% (v2: 95.24%; +0.5 pt ≈ 95.74%).
  2. No focus letter (b/d/p/q, word-mode case-merged) drops >1 pt vs v2 (floors: b ≥ 95.5%, d ≥ 97.3%, p ≥ 97.3%, q ≥ 89.8%).
  3. int8 ONNX ≤ 0.5 MB.
- **Compute:** Google Colab GPU via the MCP route (driver `colab_v4_job.py`), EMNIST downloaded on the runtime from the official source via torchvision. Launch precondition: no other job's runtime active. Budget cap: 8 compute units for this job.
- **Terms compliance (v4):** training data, labels, and filtering come **only** from EMNIST (NIST SD19, public-domain US government work; Cohen et al. 2017), same as v1–v3. **No** training data, labels, scores, rankings, or filtering from Claude/Muse or OpenAI/Luna outputs at any stage. Augmentation and oversampling are deterministic code operations on real EMNIST samples. Nothing published or uploaded anywhere.

## Results — full official EMNIST byclass test split (116,323 samples), single evaluation
- **Raw 62-class accuracy: 85.5%** (v2: 85.5%, +0.06 pts — a tie in practice)
- **Word-mode letters accuracy: 95.2%** (v2: 95.2%, **−0.04 pts**)
  - Word-mode by letter (case-merged): **b 97.1% (+0.5) · d 97.1% (−1.1) · p 98.5% (+0.2) · q 92.3% (+1.4)**
  - Word-mode, lowercase truth only: b 96.1% (+0.3) · d 97.7% (−1.2) · p 95.7% (−0.3) · q 87.9% (+3.2 — best q yet)
- Raw-mode focus (lowercase truth): b 88.7%, d 97.3%, p 15.2% (80.4% predicted as 'P', same case ambiguity as v2), q 50.1% (still 37.0% to '9', masked in word-mode).
- Selected checkpoint: epoch 12 of 12 (best val word-mode 0.9538 on the 4,960-sample carve). Training took 520 s in-kernel on a T4. Oversampling extras per epoch: q +2,914, g +1,804 (subtrain counts after the val carve).
- **Keeper gates (pre-registered above): FAILED — v4 is NOT the keeper; v2 remains the current model.**
  1. Word-mode ≥ 95.7% → **95.19% ✗**
  2. No focus letter drops >1 pt → **d −1.14 pts ✗** (b, p, q all improved)
  3. int8 ONNX ≤ 0.5 MB → **266,759 B ✓**
- Interpretation: the q/g oversampling lever worked for the third version in a row — q is now the best it has ever been on both word-mode views — and reverting to v2-strength augmentation restored the raw accuracy v3 lost (85.5% vs v3's 84.0%). But the package nets out to a wash on the headline metric: q's gain was offset by d slipping ~1.1 pts. v4 ≈ v2 with a different error distribution, not an improvement. No re-tuning was done against this spent test evaluation.

## Compute / infrastructure record (v4)
- **Colab cost: 0.71 compute units total** (balance 597.08 → 596.37; cap was 8). Two attempts were needed, both on T4 runtimes, each deleted after use; final state verified: 0 active assignments.
- Attempt 1: training completed all 12 epochs (epoch 12 selected, val word-mode 0.9534), but the local driver (`colab_v4_job.py`) crashed while polling — the Colab MCP server embeds literal newlines inside progress-notification messages, and `colab_mcp.py`'s line-splitting SSE parser died on the resulting unterminated JSON string. The driver's finally block deleted the runtime before any artifact download; **no test metric from attempt 1 was ever observed, and no decision was influenced by it.** The parser was fixed (unparseable SSE lines are skipped) and the identical pre-registered job re-run once.
- Attempt 2: training reproduced attempt 1's trajectory almost exactly (same seeds; epoch 12 selected, val word-mode 0.9538) and completed the single test evaluation + ONNX export on the runtime. The local driver process was killed externally (SIGTERM) mid-poll, so the artifacts were recovered manually through a fresh session on the still-running runtime: paired int8 check run on the runtime, all files downloaded via connectionInfo, runtime then deleted and verified gone. **The metrics above are attempt 2's single evaluation.**
- **Export note:** the runtime's newer torch exported fp32 ONNX in external-data format (graph 19,798 B + `.data` sidecar 992,384 B). The pair was downloaded and re-saved locally as a single inlined file (1,011,701 B); equivalence verified — max logit difference 0.0 on random inputs, and the paired subset check re-run locally against the final files reproduces the runtime's numbers exactly.

## Artifacts (v4, in this folder)
- `write_it_right_v4_int8.onnx` — dynamic int8 ONNX, **266,759 bytes (~0.25 MB)**
- `write_it_right_v4.onnx` — fp32 ONNX, single file, 1,011,701 bytes · `model_v4_best.pt` — selected checkpoint (epoch 12) · `results_v4.json` — full metrics (incl. infra note) · `train_v4.py` — training code · `colab_v4_job.py` — Colab driver · `train_v4.log` — runtime training log · `colab_v4_job.log` / `colab_v4_job_run2.log` — driver logs for attempts 1 and 2
- **Int8 verification** (paired, first 10,000 official test samples, `onnx_eval_v4_subset.json` via `eval_v4_int8.py`, run on the runtime and reproduced locally): int8 raw 85.29% / word-mode 94.89% vs fp32 raw 85.27% / word-mode 94.79% on the same samples — quantization cost ≈ 0, same pattern as v2/v3.

## Known limits (v4)
- Same as v2/v3: EMNIST is adult handwriting — the children's-handwriting dataset audit is still required before grading real kids' writing.
- The remaining word-mode errors are increasingly a trade-off surface between confusable letters (pushing q up cost d slightly). A future v5 would need a different lever than oversampling alone — e.g. a brief fine-tune phase at low LR, or per-letter threshold-free margin training — and its own fresh pre-registration; this test evaluation is spent.

---

# v5 — symmetric focus oversampling (2026-11-18)
Status: local iteration only — NOT published or uploaded anywhere.

## Pre-registration (written BEFORE training)
- **Base recipe: v4's exact recipe, unchanged.** v2's architecture (CNN 32/64/128/128, 248,958 params); v2-strength affine augmentation only (rotation ±8°, shift ±2 px, scale 0.95–1.05, every sample; no elastic, no flips); full official EMNIST byclass train split (697,932) minus the same small stratified validation carve (80/class = 4,960, seed 20261119) used ONLY for checkpoint selection (best val word-mode); 12 epochs, AdamW (lr 2e-3, cosine), batch 1024, seed 42. EMNIST downloaded on the runtime from the official source via torchvision.
- **THE ONE CHANGE vs v4: the focus oversampling is made symmetric.** v4 oversampled lowercase 'q' ×2 and lowercase 'g' +50% per epoch. v5 keeps those exactly and adds lowercase 'd' +50% and lowercase 'b' +50% extra copies per epoch (same every-other-sample mechanism as 'g'). Plain oversampling of real EMNIST samples; no threshold biasing, no synthetic data.
  - **Reasoning (named in advance):** three data points isolate the asymmetric q/g emphasis as the likely source of the d tax. v2 (no oversampling): d word-mode 98.25%. v3 (q/g oversampling inside a confounded package — elastic augmentation, larger net, 31k val carve): d 97.8% (−0.4). v4 (q/g oversampling as the ONLY change vs v2): d 97.12% (−1.14, failing the no-drop gate) while q rose to its best-ever 92.3%. With v4 as the clean controlled comparison, the oversampling package itself is the prime suspect: extra gradient mass on the q/g cluster shifts shared stem-and-bowl stroke features and effective class priors, and b — d's mirror partner — received no compensating emphasis. Balancing the emphasis across both confusable pairs (q/g AND b/d) is the single targeted change predicted to keep q's bought gains without skewing the b/d boundary.
- **Evaluation discipline:** the official test split is evaluated **exactly once**, on the selected checkpoint (raw + word-mode + b/d/p/q + lowercase-truth focus), followed by the paired int8-vs-fp32 subset check on the first 10,000 test samples, as in v2–v4.
- **Keeper gates (unchanged from v4, ALL must pass, else v5 is rejected and v2 stays):**
  1. Word-mode letters accuracy ≥ 95.7% (v2: 95.24%).
  2. No focus letter (b/d/p/q, word-mode case-merged) drops >1 pt vs v2 (floors: b ≥ 95.5%, d ≥ 97.3%, p ≥ 97.3%, q ≥ 89.8%).
  3. int8 ONNX ≤ 0.5 MB.
- **Compute:** Google Colab GPU via the MCP route (driver `colab_v5_job.py`), own per-job runtime, deleted in a finally block. Launch precondition: no other job's runtime/assignment active. Budget cap: 8 compute units for this job.
- **Terms compliance (v5):** training data, labels, and filtering come **only** from EMNIST (NIST SD19, public-domain US government work; Cohen et al. 2017), same as v1–v4. **No** training data, labels, scores, rankings, or filtering from Claude/Muse or OpenAI/Luna outputs at any stage. Augmentation and oversampling are deterministic code operations on real EMNIST samples. Nothing published or uploaded anywhere.

## Results — full official EMNIST byclass test split (116,323 samples), single evaluation
- **Raw 62-class accuracy: 85.3%** (v2: 85.5%, −0.2 pts; v4: 85.5%)
- **Word-mode letters accuracy: 95.2%** (v2: 95.24%, **−0.06 pts**; v4: 95.19%)
  - Word-mode by letter (case-merged): **b 97.2% (+0.7) · d 97.6% (−0.6) · p 98.1% (−0.2) · q 91.2% (+0.3)** (deltas vs v2)
  - Word-mode, lowercase truth only: b 96.2% (+0.4) · d 98.5% (−0.4) · p 94.6% (−1.4) · q 86.7% (+2.0)
- Raw-mode focus (lowercase truth): b 88.9%, d 98.1%, p 17.1% (77.4% predicted as 'P', same case ambiguity as always), q 49.9% (still 37.0% to '9', masked in word-mode).
- Selected checkpoint: epoch 10 of 12 (best val word-mode 0.9546 on the 4,960-sample carve — the best validation score of the whole v2-recipe line; v4: 0.9538). Training took 508 s in-kernel on a T4. Oversampling extras per epoch: q +2,914, g +1,804, d +5,049, b +2,540.
- **Keeper gates (pre-registered above): FAILED on gate 1 only — v5 is NOT the keeper; v2 remains the current model.**
  1. Word-mode ≥ 95.7% → **95.18% ✗**
  2. No focus letter drops >1 pt → **✓ PASSES** (b +0.67, d −0.65, p −0.17, q +0.33 — all inside their floors; first time an oversampling version passes this gate)
  3. int8 ONNX ≤ 0.5 MB → **266,759 B ✓**
- Interpretation: the pre-registered hypothesis was HALF right. Symmetric oversampling did recover most of the d tax — d went from v4's −1.14 (gate breach) to −0.65, and d lowercase-truth from −1.19 to −0.42 — confirming the asymmetric q/g emphasis was indeed taxing d. But the cost moved rather than vanished: q gave back most of v4's gain (92.3% → 91.2% case-merged; 87.9% → 86.7% lowercase-truth) as the added b/d copies (~7,600/epoch) diluted q/g's relative emphasis. Net headline: 95.18%, statistically a three-way tie with v2 (95.24%) and v4 (95.19%). Across three controlled data points (v3 confounded, v4 clean, v5 clean), oversampling — symmetric or not — only REDISTRIBUTES errors among the confusable letters; it does not create accuracy. This recipe's word-mode ceiling on EMNIST sits at ≈95.2%, and the remaining errors are a zero-sum surface between the q/g and b/d pairs plus genuinely ambiguous glyphs. Breaking 95.7% will need a different lever (capacity, loss shaping, longer/different training), not a fourth oversampling permutation. No re-tuning was done against this spent test evaluation.

## Compute / infrastructure record (v5)
- **Colab cost: 0.28 compute units** (balance 595.06 → 594.78; cap was 8). Launch waited ~70 min for the Scam Shield v7 / Read-Aloud v3 jobs to clear (precondition: no active assignments), then ran on a single T4 runtime, deleted afterwards; final state verified: 0 active assignments.
- One driver attempt. Training completed cleanly on the runtime, but mid-run the MCP execute_code channel began returning HTTP 200 with no parseable result and the local driver's polls failed persistently. The driver was stopped; all artifacts were recovered through a fresh session + connectionInfo download (the v4 attempt-2 recovery pattern), **verified non-empty locally before the runtime was deleted**. The metrics above are the training script's own single test evaluation (printed in `train_v5.log`, stored by the script in `results_v5.json`).
- **Export note:** the runtime exported fp32 ONNX in external-data format (graph 19,798 B + `.data` sidecar 992,384 B). The pair was downloaded and re-saved locally as a single inlined file (1,011,701 B); equivalence verified — max logit difference 0.0 on random inputs. The `.data` sidecar was removed after inlining.
- **Int8 verification** (paired, first 10,000 official test samples, `onnx_eval_v5_subset.json` via `eval_v5_int8.py`, run locally against the final files): int8 raw 85.03% / word-mode 95.02% vs fp32 raw 85.18% / word-mode 95.10% on the same samples — quantization cost ≈ 0, same pattern as v2–v4.

## Artifacts (v5, in this folder)
- `write_it_right_v5_int8.onnx` — dynamic int8 ONNX, **266,759 bytes (~0.25 MB)**
- `write_it_right_v5.onnx` — fp32 ONNX, single file, 1,011,701 bytes · `model_v5_best.pt` — selected checkpoint (epoch 10) · `results_v5.json` — full metrics (incl. verdict + infra note) · `train_v5.py` — training code · `colab_v5_job.py` — Colab driver · `train_v5.log` — runtime training log · `colab_v5_job.log` — driver log

## Known limits (v5)
- Same as v2–v4: EMNIST is adult handwriting — the children's-handwriting dataset audit is still required before grading real kids' writing.
- The oversampling lever is now measured to exhaustion (v3/v4/v5): it trades q against d roughly one-for-one around a ≈95.2% word-mode ceiling for this architecture/recipe. Any v6 should change a different variable entirely and carry its own fresh pre-registration; this test evaluation is spent.

---

# v6 — shear augmentation + longer training (2026-11-18) — ATTEMPT 1 BLOCKED AT PRECONDITION
Status: NOT RUN. No diagnosis, no training, no test evaluation, **0 Colab units spent by this job**.

- The launch precondition (no other job's runtime/assignment active — Read-Aloud v3 and Scam Shield v8 were in flight, plus other chats' jobs on the shared account) never cleared during ~3 hours of polling at ~5-min intervals. Observed runtime count over the window: 1 → 2 → 1 → 2 → 3 → 2; at the final check, 2 runtimes were still active (one T4 that stayed alive for the entire ~3 h window, plus an L4) with the account burning 3.09 units/hr from other jobs. Balance moved 594.76 → 592.16 entirely due to other jobs; this job created no runtime.
- Everything is staged for an immediate retry when the account clears:
  - `diagnose_v6.py` — Step-1 diagnosis (keeper v2 ONNX on EMNIST TRAIN slices only: the v4/v5 4,960-sample carve for per-letter word-mode + 'a' confusion partners, a TRAINREF slice as the capacity probe, and a projection-profile slant analysis with accuracy by shear bucket). The official test split is never touched by it.
  - `colab_v6_diag_job.py` — Phase-A driver: guard → runtime → setup → ships diagnose_v6.py + keeper ONNX (base64-chunked, size-asserted) → downloads diagnosis_v6.json verified non-empty → deletes runtime in finally.
  - `train_v6.py` — the v6 trainer per the task design: v2 base recipe, NO oversampling, v2 affine + horizontal shear ±0.25, 16 epochs, width switch (narrow 32/64/128/128 vs wide 40/80/160/160) to be set from the Step-1 capacity probe and recorded in the pre-registration BEFORE training; final single test evaluation reports a/b/d/p/q.
  - `eval_v6_int8.py` — paired int8-vs-fp32 subset check (v6 filenames).
  - Still owed on retry, in order: run diagnosis → record findings here → write the v6 pre-registration (gates: word-mode ≥95.7%; no b/d/p/q drop >1 pt vs v2; int8 ≤0.5 MB; 'a' reported, no gate) → write colab_v6_job.py (Phase-B driver, modeled on colab_v5_job.py) → train.
- Terms compliance: nothing ran, so nothing to affirm beyond the standing rule — EMNIST only, no frontier-model output in data/labels/filtering, nothing published.

---

# v6 — RETRY (2026-11-18/19): Step-1 diagnosis results + PRE-REGISTRATION (written BEFORE v6 training)

Sequencing note: attempt 1 was blocked by the old any-runtime precondition. Rusty's updated rule (2026-11-18) — GPU jobs queue only behind GPU jobs, CPU jobs never block — unblocked the retry. Both drivers' guards were updated to GPU-only. The retry waited ~15 min for the Read-Aloud v3 T4 and another chat's L4 to clear, then ran the diagnosis on a T4 (job wall time 219 s; runtime deleted in finally; `diagnosis_v6.json` downloaded and verified non-empty BEFORE deletion).

## Step-1 diagnosis findings (`diagnosis_v6.json`; keeper v2 fp32 ONNX on EMNIST TRAIN slices only — official test untouched; v2 trained on the full train split, so absolute values are optimistic — relative structure only)
- **Per-letter word-mode (CARVE, n=160/letter):** weakest letters are **i 0.600**, g 0.906, q 0.913, l 0.925, v 0.931, j 0.938. CARVE overall word-mode 0.9575.
- **'a' is NOT weak on EMNIST:** a = **0.9813** case-merged, **0.9625** lowercase-only. Its confusion partners are exactly the ones Rusty's handwriting hits: **q is the #1 partner** (1.25% of merged a; 2.5% of lowercase a), then c (1.25% of lowercase a). Interpretation: Rusty's 4-of-5 failure is a style/distribution gap, not an EMNIST-wide 'a' weakness — his left-handed single-story 'a' with a long right tail sits on the a/q boundary the diagnosis names, and EMNIST 'a' exemplars are dominated by forms unlike his. Shear augmentation is the lever aimed at that boundary's shape variation; whether it moves HIS 'a' can only be checked with his own samples (the kids'-tool chat's territory), not with EMNIST.
- **Capacity probe:** CARVE 0.95745 vs TRAINREF (samples v2 trained on for 10 epochs) 0.95841 — gap **0.10 pts**. No memorisation/generalisation gap: capacity does **not** bind for this recipe.
- **Slant/shear buckets (word-mode):** near-zero shear (−0.1..0.1) 0.9619; −0.2..−0.1 0.9661; <−0.2 0.9557; **positive-shear buckets are the weakest: 0.1..0.2 → 0.9442, >0.2 → 0.9507**. Population mean shear −0.16, std 0.27 — slant variation is wide. 'a' itself is slant-insensitive on EMNIST (0.956–1.000 in every bucket) but has the widest shear spread of any letter (std 0.311 vs 0.265 overall). The ±0.25 shear augmentation covers exactly the weak positive-shear range.

## PRE-REGISTRATION (locked before training)
- **Width decision: NARROW — 32/64/128/128 (248,958 params, v2's architecture).** The capacity probe shows no capacity binding (gap 0.10 pts), so the pre-registered condition for widening (40/80/160/160) is not met. (v3's wider net also regressed, albeit confounded.)
- **Recipe:** v2 base recipe unchanged except the two pre-registered levers — (1) augmentation = v2 affine (rot ±8°, shift ±2 px, scale 0.95–1.05) **plus horizontal shear ±0.25 rad**; (2) **16 epochs** (v2: 10). NO oversampling of any class. Full train split minus the 4,960-sample stratified val carve (seed 20261119) used only for checkpoint selection (best val word-mode). AdamW lr 2e-3 cosine, batch 1024, seed 42.
- **Evaluation discipline:** official test split evaluated ONCE, at the end, on the selected checkpoint (raw + word-mode + a/b/d/p/q + lowercase-truth), then ONNX export + paired int8-vs-fp32 subset check (first 10,000 test samples), as in v2–v5.
- **Keeper gates (ALL must pass or v6 is rejected and v2 stays keeper):**
  1. Word-mode ≥ **95.7%** (v2: 95.24%).
  2. No b/d/p/q drop > 1 pt vs v2 (floors: b ≥ 95.5, d ≥ 97.3, p ≥ 97.3, q ≥ 89.8).
  3. int8 ONNX ≤ 0.5 MB.
  4. 'a' word-mode reported by name — **no gate this round**; if 'a' < 90% on the test evaluation, that fact is stated plainly here and in the report.
- Terms compliance: EMNIST only (NIST SD19, public domain); deterministic code augmentation only; no frontier-model output in data/labels/filtering; nothing published.

## Results (v6, single test evaluation)
**Verdict: NOT A KEEPER — v2 remains the keeper.** Gate 1 fails; gates 2 and 3 pass.

| Metric (official test, 116,323 samples; letters n=58,405) | v6 | v2 (keeper) | Gate |
|---|---|---|---|
| Word-mode accuracy | **95.16%** | 95.24% | ≥95.7% ✗ (−0.08 pts vs v2) |
| Raw 62-class accuracy | 85.27% | 85.5% | — |
| b word-mode | 97.27% | 96.54% | ≥95.5 ✓ (+0.73) |
| d word-mode | 97.44% | 98.25% | ≥97.3 ✓ (−0.81, inside the 1-pt allowance) |
| p word-mode | 98.30% | 98.30% | ≥97.3 ✓ (0.00) |
| q word-mode | 91.94% | 90.85% | ≥89.8 ✓ (+1.09; second-best q of the line, after v4's 92.27%) |
| **a word-mode** | **96.30%** (n=2,706); lowercase-truth **94.53%** (n=1,644) | not recorded for v2 | reported, no gate — comfortably above 90% |
| int8 ONNX | **266,759 B** | 259,143 B | ≤0.5 MB ✓ |

- Selected checkpoint: epoch 14 of 16; best val word-mode **0.9555** — the best validation score of the v2-recipe line (v5: 0.9546) — yet the test score stayed at the ceiling. The val carve is train-distribution; the ≈95.2% test ceiling has now held against a fourth distinct lever (oversampling ×3 in v3/v4/v5, shear + longer training here).
- What the levers did: shear + 16 epochs bought real focus-letter gains (q +1.09, b +0.73) at a small d cost (−0.81) and no headline movement — the same zero-sum redistribution pattern as oversampling, now demonstrated for augmentation too. Errors move between confusable letters; they do not leave.
- **'a' report (by name, per pre-registration):** test word-mode 96.30%, lowercase-truth 94.53% — no sub-90% warning applies. Raw lowercase 'a' confusions ≥1%: Q 2.31%, A 1.52%, 2 1.28%, 9 1.16% — the a/Q boundary the diagnosis named remains 'a's top raw confusion, but in word-mode (the product metric) 'a' is one of the stronger letters. This says nothing about Rusty's own 'a' (EMNIST distribution ≠ his hand); that check belongs to the kids'-tool chat with his samples.
- **Int8 verification** (paired, first 10,000 official test samples, `onnx_eval_v6_subset.json` via `eval_v6_int8.py`, run locally against the final files): int8 word-mode 95.20% / raw 85.11% vs fp32 95.16% / 85.21% — quantization cost ≈ 0, same pattern as v2–v5.

## Compute / infrastructure record (v6)
- **Colab cost: ≈0.8 units total** (cap 8) — diagnosis ~0.07 (T4, 219 s job) + Phase B ~0.73 (T4). Shared-account caveat: other jobs were active during the early wait window; the Phase-B window delta (591.53 → 590.80) also includes a CPU job at 0.08 units/hr. Final state verified: **0 active assignments, burn 0.00/hr**.
- Two infra incidents, both recovered without touching the evaluation's integrity: (1) the Phase-B driver process was SIGTERM-killed ~7 min in (epoch 6) — training runs detached on the runtime and was found alive at epoch 8 by the recovery driver (`colab_v6_recover.py`, the v4/v5 adopt-the-runtime pattern); (2) the recovery's 30-min poll missed the DONE flag by a hair, so the int8 subset check was run locally against the final files instead (v5 precedent) and the runtime — left alive by the recovery's incomplete-artifacts guard — was deleted manually once every artifact was verified local. Roughly ~0.3 units of the Phase-B spend was idle burn from these two incidents.
- **Export note:** the runtime exported fp32 ONNX in external-data format (graph 19,798 B + `.data` sidecar 992,384 B); the pair was downloaded and re-saved locally as a single inlined file (1,011,701 B); equivalence verified — max logit difference 0.0 on random inputs. The `.data` sidecar was removed after inlining.

## Artifacts (v6, in this folder)
- `diagnosis_v6.json` — Step-1 diagnosis · `write_it_right_v6_int8.onnx` — dynamic int8 ONNX, **266,759 bytes (~0.25 MB)** · `write_it_right_v6.onnx` — fp32 ONNX, single file, 1,011,701 bytes · `model_v6_best.pt` — selected checkpoint (epoch 14) · `results_v6.json` — full metrics (incl. gates + verdict) · `train_v6.py` — training code · `diagnose_v6.py` — diagnosis code · `colab_v6_diag_job.py` / `colab_v6_job.py` / `colab_v6_recover.py` — Colab drivers · `train_v6.log` — runtime training log · `colab_v6_diag_run.log` / `colab_v6_job.log` / `colab_v6_recover.log` — driver logs

## Known limits (v6)
- Same as v2–v5: EMNIST is adult handwriting — the children's-handwriting dataset audit is still required before grading real kids' writing.
- The ≈95.2% word-mode ceiling for this architecture/recipe family is now supported by four controlled data points (v3 confounded; v4, v5, v6 clean). Any v7 should change the architecture or the loss, not another data/augmentation permutation — and would need its own fresh pre-registration. This test evaluation is spent.

---

# v7 — architecture-or-loss change (2026-11-19) — PRE-REGISTRATION (written BEFORE v7 training and BEFORE any v7 test evaluation)

Standing ruling carried in from v6: v7 must change the architecture or the loss — no more data/augmentation permutations. Two candidate paths were specified in advance; the sanity check below selects exactly ONE, and only that path is trained and evaluated on the official test (once).

## Path A (primary, as designed): two-stage cascade
- Stage 1 = the frozen v2 keeper network, untouched (weights `model_full_ep10.pt`).
- Stage 2 = small specialist classifiers for the confusable groups {a,q,g} and {b,d,p}, invoked only when stage 1's top-2 case-merged letter predictions both fall inside one group; the specialist's verdict replaces stage 1's within that group. Specialists to be trained ONLY on EMNIST train-split samples (deployment-matched triggered subset), word-mode labels. Total deployed int8 (stage-1 v2 int8 259,143 B + specialists) ≤ 0.5 MB.

## Sanity check for Path A (`sanity_v7.py` → `sanity_v7.json`; LOCAL, TRAIN SPLIT ONLY — official test untouched)
Method: v2 weights loaded into the v2 architecture; pipeline verified by exact reproduction of the v6 diagnosis carve figure (stage-1 carve word-mode **0.95745** = diagnosis_v6's 0.9574519230769231). Word-mode top-2 = top-2 distinct letters by per-letter max-over-cases logit. Slices: the v4/v5/v6 val carve (4,160 letters) and a uniform random 40,000-row train slice (20,249 letters; the better structural proxy for the natural-distribution test set). v2 trained on all of the train split, so absolute accuracies are optimistic — trigger/capture structure only.

| Quantity | Carve | Random 40k train slice |
|---|---|---|
| Stage-1 word-mode | 0.9575 | 0.9556 |
| Trigger rate (top-2 within one group) | 9.0% | 5.6% |
| Addressable errors (triggered ∧ wrong ∧ truth in group) | 30 of 177 errors | 77 of 899 errors (8.6%) |
| Breakable correct (triggered ∧ stage-1 already right) | 345 | 1,045 |
| **Oracle gain (PERFECT specialist)** | **+0.72 pts** | **+0.38 pts** |
| Estimated triggered train samples | — | {a,q,g} ≈ 14,447 · {b,d,p} ≈ 5,234 |

**Soundness criterion (fixed before the check was run):** the cascade is sound only if the perfect-specialist oracle gain on the representative slice is at least ~2× the +0.46 pts the keeper gate requires (95.24% → 95.70%), i.e. ≥ +0.9 pts, with ≥ 3,000 triggered training samples per group.
**Verdict: Path A is UNSOUND — rejected before training.** The oracle gain on the representative slice (+0.38 pts) is below even the +0.46 pts a *perfect* specialist would need; the {b,d,p} group contributes almost nothing (7 of 77 addressable errors), and the breakable-to-addressable ratio is ~14:1 (a realistic specialist at ≤97% on the triggered subset nets ≈ 0 or negative after breakage). Data volume was not the problem (both groups exceed 3,000 triggered train samples). The cascade's trigger simply does not cover the errors that make up the ceiling — most word-mode errors have their runner-up outside the confusable group.

## Path B (fallback, CHOSEN): loss change — focal loss, full retrain
Recorded here before training, per the fallback rule. **v7 = Path B.**
- **Architecture: v2's, unchanged** — narrow CNN 32/64/128/128, 248,958 params, 62 classes.
- **Recipe: v2's, unchanged in every other respect** — full train split (697,932) minus the same 4,960-sample stratified val carve (80/class, seed 20261119) used only for checkpoint selection (best val word-mode); v2 affine augmentation only (rot ±8°, shift ±2 px, scale 0.95–1.05; NO shear, NO oversampling); AdamW lr 2e-3, weight_decay 1e-4, cosine schedule, batch 1024, seed 42, **10 epochs**.
- **The single change: the loss.** Focal loss with γ = 2.0 replaces cross-entropy: per-sample loss = (1 − p_t)^2 · CE, mean over batch. Rationale: the ceiling errors are a thin hard tail that plain CE barely weights once the easy majority is fit; focal loss continuously up-weights exactly those hard/confusable samples without duplicating classes (the oversampling failure mode) or touching data/augmentation.
- **Evaluation discipline:** official test split evaluated ONCE, at the end, on the selected checkpoint (raw + word-mode + a/b/d/p/q + lowercase-truth + the a→q figure below), then ONNX export + paired int8-vs-fp32 subset check (first 10,000 test samples), as in v2–v6.
- **Keeper gates (ALL must pass or v7 is rejected and v2 stays keeper):**
  1. Word-mode ≥ **95.7%** (v2: 95.24%).
  2. No b/d/p/q drop > 1 pt vs v2 (floors: b ≥ 95.54, d ≥ 97.25, p ≥ 97.30, q ≥ 89.85).
  3. Total deployed int8 ≤ 0.5 MB (single network; v2's int8 was 259,143 B).
  4. 'a' word-mode reported by name, **plus the a→q confusion rate**: among official-test samples whose true class is lowercase 'a', the fraction whose word-mode prediction is 'q' — the test-split equivalent of the v6 diagnosis train-slice figure of **2.5%** (carve, lowercase a → q, word-mode). No gate on 'a' itself; reported plainly either way.
- **Compute:** Colab **CPU** runtime (Rusty's 2026-11-18 sequencing: GPU jobs queue behind Read-Aloud v4's GPU job; CPU work runs in parallel and does not block). One tar bundle up, one tar bundle down. Runtime deleted in a finally block; downloads verified non-empty locally BEFORE deletion; list_runtimes + balance verified after. Cap 8 units.
- Terms compliance: EMNIST only (NIST SD19, public domain); deterministic code augmentation only; no frontier-model output in data/labels/filtering; nothing published.



## Results (v7, single test evaluation, focal-loss path)

Executed exactly as pre-registered: focal loss gamma=2.0, v2 recipe
otherwise (narrow 32/64/128/128, 248,958 params, v2 affine only, NO
shear, NO oversampling), full train split minus the same 4,960-sample
carve (seed 20261119), AdamW lr 2e-3 cosine, batch 1024, seed 42,
10 epochs, best-validation checkpoint selection. Ran on a Colab CPU
runtime (the GPU queue stayed occupied by other jobs the whole
window). Training wall time 23,859 s (~39.7 min/epoch at the CPU
runtime's effective ~1 core). Selected checkpoint: epoch 9, val
word-mode 0.9526. The official test was evaluated exactly ONCE
(116,323 samples). Full numbers: results_v7.json.

| Metric (official test) | v7 focal | v2 keeper | Gate | Pass |
|---|---|---|---|---|
| Word-mode (letters) | 0.9518 | 0.9524 | >= 0.957 | NO |
| Raw 62-class | 0.8519 | 0.855 | -- | -- |
| b word-mode | 0.9694 | 0.9654 | >= 0.9554 | yes |
| d word-mode | 0.9789 | 0.9825 | >= 0.9725 | yes |
| p word-mode | 0.9824 | 0.9830 | >= 0.9730 | yes |
| q word-mode | 0.8922 | 0.9085 | >= 0.8985 | NO (-1.63 pts) |
| int8 size | 266,759 B | 259,143 B | <= 0.5 MB | yes |

'a' word-mode on test: 0.9660 (merged, n=2,706); lowercase-truth
0.9483 (n=1,644). **a->q confusion (word-mode, lowercase-true 'a'
predicted 'q'): 2.19% on the test split** (36/1,644), vs the v6
diagnosis figure of 2.5% measured on a held-out TRAIN slice (same
quantity, different split) -- a small improvement in the target
confusion; merged-a->q is 1.33%. But focal bought it by moving mass
inside the a/q/g cluster: q itself fell 1.63 pts, the largest
single-letter regression of any v3-v7 candidate, and overall
word-mode did not move.

Int8 check (local, first 10,000 test samples,
onnx_eval_v7_subset.json): fp32 word-mode 0.9530 vs int8 0.9522 --
quantization cost ~0.08 pts, same magnitude as v2's.

**Verdict: NOT KEEPER (1 of 3 gates). v2 remains the keeper.** The
~95.2% word-mode ceiling has now held against every lever tried:
oversampling in three variants (v3/v4/v5), shear augmentation +
longer training (v6), a cascade design measured unsound on
train-split arithmetic before any training (v7 Path A), and a loss
change (v7 Path B, focal). The recipe-faithful test evaluations
cluster within 0.08 pts of each other (95.16-95.24%). The remaining
errors look like EMNIST label/shape ambiguity at this model scale
rather than a fixable training defect; further gains likely need a
different data source (e.g. real children's handwriting, still
unaudited) or a materially different model class, not another
permutation of this recipe.

Cost: ~0.6 units attributable to v7 (CPU runtime ~7.1 hr at
~0.08/hr; the account is shared and other jobs' GPU time dominates
the window's total balance delta). Runtime deleted and verified
via list_runtimes; final balance 581.91. Infra notes: the local
driver process was killed by three separate VM restarts and
relaunched in adopt mode each time (runtime and detached training
unaffected); near the end the execute_code channel degraded
(HTTP 200, no parseable result -- the v5 failure mode), so the
finished artifacts were recovered via the connectionInfo download
path and verified locally BEFORE the runtime was deleted, per the
standing rule.

---

# C3 — stroke follow-through (2026-11-19) — PRE-REGISTRATION (written BEFORE any C3 training and BEFORE any C3 exam evaluation)

## Trigger (from LINEUP-2026-11-19.md, verified against stroke-exp/results.json)
The stroke experiment's pre-registered criterion passed: on the shared UJIpenchars2
test (2,080 letters, 20 unseen writers) the real-stroke model (Arm 1) beat the
image keeper v2 (Arm 2) — overall 85.19% vs 82.12% at ~27 KB vs 259 KB int8;
combined pair error a/q 10.6% vs 24.4%, e/f 13.1% vs 28.1%, n/h 23.1% vs 23.8%
(a wash); directed a->q 10.0% -> 0.0%. Trajectory recovery from images does NOT
transfer (Arm 3: 51.9%) — **no recovered/EMNIST trajectories are used as C3
training data.** C3 therefore runs. The G2 hybrid cascade is a separate later
step and is NOT built here.

## The one lever
Training data only. Architecture (stroke-exp/train_stroke.py StrokeNet,
27,194 params), featurization (stroke-exp/common.py featurize, 5x64), training
protocol (Adam lr 1e-3, batch 64, max 60 epochs, early stop patience 10 on val,
seed 0) are IDENTICAL for both models below. Two models are trained:
- **CONTROL** — Arm 1 recipe exactly: UJI training writers only.
- **C3** — CONTROL data **plus** (a) Hershey-font synthetic strokes with frozen
  instance perturbation, and (b) handedness augmentation. Nothing else changes.

## Data
- **UJIpenchars2** (CC BY 4.0, UCI dataset 177 — as logged in stroke-exp/DESIGN.md).
  All 60 writers pooled and RE-PARTITIONED for C3 (see Exam). Training pool =
  the 40 writers NOT in the C3 exam. Validation = training-pool writers with
  writer id % 7 == 0 (ids 7, 14, 28, 35, 49, 56; 624 samples), identical for
  CONTROL and C3; checkpoint selection on this val only.
- **Hershey fonts — license verified at source 2026-11-19.** Created by
  Dr. A. V. Hershey at the U.S. National Bureau of Standards; public domain.
  The Usenet distribution (Ghostscript Hershey.htm, plan9.io mirror) states the
  fonts "are in the public domain" and this distribution "may be used by anyone
  for any purpose, commercial or otherwise", with acknowledgements to Hershey
  and to James Hurt (format) and the restriction that the data not be converted
  to the NTIS distribution format (it is not). Glyphs accessed via the
  `pyhershey` 0.2.0 package database (roman_simplex + script_simplex mappings).
  Acknowledgement: Hershey Fonts originally created by Dr. A. V. Hershey, U.S.
  National Bureau of Standards; distribution format by James Hurt, Cognition Inc.
- **Hershey synthetic generation (frozen, deterministic code, seed 20261119):**
  glyphs = ASCII A-Z and a-z (52) x fonts {roman_simplex, script_simplex};
  40 instances per glyph per font = 4,160 synthetic samples, case-merged labels.
  Hershey coordinates are y-up; y is flipped to the UJI y-down convention.
  Per-instance perturbation (this is how single glyphs become instances; part
  of the synthetic-data lever, frozen here): uniform scale 0.85-1.15, slant
  shear k in [-0.12, +0.12] (x += k*(y-0.5) in unit box), rotation +/-6 deg,
  per-point Gaussian jitter sigma 0.01 in unit-box coordinates.
- **Handedness augmentation (frozen):** applied to a fixed 25% of UJI training
  samples (seeded selection) and 25% of Hershey instances, replacing the
  original. Transform, in unit-box coordinates: (1) every predominantly
  horizontal stroke (x-range > 2 * y-range) has its point order reversed;
  (2) backslant shear x -= 0.30 * (0.5 - y) (top of the letter shifts left).
  Targets the left-handed single-story 'a' and reversed horizontal strokes.
- **UNIPEN: NOT used** (research-eval only, license conflict unresolved).
  No frontier-model output in data, labels, filtering, or selection anywhere.

## Fresh exam (frozen, evaluated ONCE per model at the end)
- **Writers:** UJI writer ids divisible by 3: 3, 6, 9, 12, 15, 18, 21, 24, 27,
  30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60 (20 writers x 104 = 2,080 letter
  samples). Neither CONTROL nor C3 trains on any exam writer; val writers are
  disjoint from exam writers. **Independence caveat, stated plainly:** this is
  a NEW split for C3, but it is not virgin data for the project — 17 of these
  writers (12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57,
  60) served as the stroke experiment's test (tst) set and the other 3
  (3, 6, 9) were in its training (trn) pool. (Wording corrected before any
  C3 training; the split and gates were unchanged.) C3's models have never trained on any of them, and no C3
  design choice was tuned on the experiment's per-writer results, but the
  85.2% Arm 1 headline came from a subset of these writers — so Arm 1's 85.2%
  is a descriptive benchmark only, NOT a gate. The gates below compare C3 to
  CONTROL and to the image keeper on THIS exam, same samples, same sitting.
- **Image keeper on this exam:** keeper v2 int8 scored on renders of the same
  2,080 exam samples through the demo preprocessing pipeline (Arm 2 method).
- **Adversarial probes (built from exam samples only, never trained on):**
  - P1 left-handed 'a': all exam 'a' samples (n=80) with the handedness
    transform above applied. Named target: left-handed-style 'a'.
  - P2 reversed direction: all exam e/f/n/h samples (n=320) with every
    stroke's point order reversed. Named target: Arm 1's residual
    reverse-direction confusions (e->f, h->n).

## Keeper gates (ALL must pass or C3 is rejected; frozen)
1. Headline: C3 exam accuracy >= CONTROL exam accuracy AND >= image keeper v2
   accuracy on the same exam samples AND >= 0.82 absolute.
2. Pairs: C3 sum of combined pair errors (a/q + e/f + n/h) <= CONTROL's sum,
   AND C3 primary directed-confusion sum <= CONTROL's.
3. a->q: C3 exam a->q rate <= CONTROL's.
4. Probes: P1 C3 accuracy >= CONTROL's; P2 C3 accuracy >= CONTROL's - 0.02
   (2-pt non-regression tolerance = ~6 of 320 samples; direction reversal is
   out-of-distribution for both models and P2 is a non-regression check).
5. Size: deployed size <= 0.5 MB (int8 estimate = parameter count in bytes,
   expected ~27 KB; fp32 state dict expected ~115 KB — both reported).

## Compute
Local CPU only (tools-venv, torch 2.7.1). **Colab units: 0.** Nothing
published or pushed. Artifacts under `write-it-right/stroke-v2/`.

## Results (C3, single exam evaluation per model, as pre-registered)

Executed exactly as pre-registered: identical StrokeNet/featurization/protocol
for both models; CONTROL = UJI training pool only (3,536 train / 624 val);
C3 = CONTROL data with handedness replacement on 25% of UJI train samples
plus 4,160 Hershey synthetic instances (roman_simplex + script_simplex,
40/glyph/font, frozen perturbations, 25% handedness-transformed). Exam =
2,080 samples from the 20 writers with id % 3 == 0, evaluated ONCE per model.
CONTROL early-stopped at epoch 50 (val best 0.9215); C3 at epoch 38
(val best 0.9054). Image keeper v2 scored 0.8245 on renders of the same exam
samples (a->q 8.75%). Full numbers: `stroke-v2/results_c3.json`.

| Metric (exam) | CONTROL | C3 | Keeper v2 (same samples) |
|---|---|---|---|
| Overall accuracy | **0.8692** | 0.8582 | 0.8245 |
| a/q combined err | 0.1125 | 0.1062 | 0.2500 |
| e/f combined err | 0.1187 | 0.1437 | 0.2938 |
| n/h combined err | 0.2313 | **0.1688** | 0.2750 |
| Combined pair sum | 0.4625 | **0.4188** | 0.8187 |
| Primary directed sum | **0.0750** | 0.1375 | 0.1125 |
| a->q | **0.0000** | 0.0375 | 0.0875 |
| e->f / f->e | 0.025 / 0.0375 | 0.000 / 0.050 | 0 / 0 |
| n->h / h->n | 0.000 / 0.0125 | 0.050 / 0.000 | 0.025 / 0 |
| P1 left-handed 'a' probe (n=80) | 0.675 | **0.825** | — |
| P2 reversed-direction e/f/n/h (n=320) | 0.1437 | 0.1375 | — |
| Size (int8 est / fp32) | 27,194 B / 114,995 B | 27,194 B / 114,995 B | 259 KB |

### Gates — 2 of 5, REJECTED
1. Headline (C3 >= CONTROL and >= keeper and >= 0.82): **FAIL** —
   0.8582 < CONTROL 0.8692 (beats keeper 0.8245 and the 0.82 floor).
2. Pairs (combined sum <= CONTROL's AND primary directed sum <= CONTROL's):
   **FAIL** — combined sum improved (0.4188 vs 0.4625) but the primary
   directed sum worsened (0.1375 vs 0.0750).
3. a->q <= CONTROL's: **FAIL** — 0.0375 vs 0.0000. The control made zero
   a->q errors on this exam; C3 reintroduced them.
4. Probes (P1 >= CONTROL; P2 >= CONTROL - 0.02): **PASS** — P1 0.825 vs
   0.675 (+15 pts on the lever's named target); P2 0.1375 vs 0.1437
   (both models are poor on fully reversed stroke direction — an
   out-of-distribution transform, as registered).
5. Size <= 0.5 MB: **PASS** (27 KB int8 estimate).

**Verdict: C3 is REJECTED — not a keeper candidate. CONTROL (the plain
Arm 1 recipe on the new split) is the better stroke model from this run.**
What the lever genuinely did: the handedness augmentation worked exactly
where aimed — left-handed-style 'a' recognition +15 points on P1, and the
n/h combined error fell from 23.1% to 16.9% (the experiment's hardest pair).
What it cost: headline accuracy (-1.1 pts), val accuracy (0.9054 vs 0.9215),
e/f combined error (11.9% -> 14.4%), and a->q errors the control never made.
Read: the synthetic Hershey block + replacing (rather than adding to) a
quarter of the real samples diluted real-writer data faster than the new
signal paid for it. A future attempt would ADD handedness copies instead of
replacing, and/or cut the Hershey share — but that is a new pre-registration,
not a C3 retry. For G2 (separate step): the stroke specialist to hybridize
should be a CONTROL-recipe model, and the P1 result says handedness
augmentation belongs in its training data as an additive block if G2's own
gates are written to protect headline and a->q.

Cost: local CPU only, **0 Colab units**. Nothing published or pushed.
Artifacts in `stroke-v2/`: prep_c3.py, control_data.npz, c3_data.npz,
probes.npz, exam_samples.pkl, control_* / c3_* (model .pt, pred .npz,
info .json), train_control.log, train_c3.log, eval_c3.py, results_c3.json.

---

# v8 — G2 stroke-specialist hybrid (2026-11-19) — PRE-REGISTRATION (written BEFORE any v8 exam evaluation)

Trigger: G2 in LINEUP-2026-11-19.md was conditional on the stroke
experiment and/or v7 justifying a stroke specialist. Both do: the stroke
experiment's real-stroke model beat the image keeper on UJI (85.2% vs
82.1%, a->q 0% vs 10%), and C3's CONTROL (plain Arm 1 recipe on a fresh
split) beat the keeper on its exam (0.8692 vs 0.8245, a->q 0.0000). C3's
own candidate was rejected and is NOT used here. v7 Path A (an
image-only cascade specialist) was measured unsound and is NOT revived.

## Design (frozen; NO training in v8)

- **Component 1 (image):** frozen keeper v2, `write_it_right_v2_int8.onnx`
  (259,143 B), scored exactly as in `eval_v2_int8.py` (letters-masked
  argmax, case-merged).
- **Component 2 (stroke):** frozen C3 CONTROL model,
  `stroke-v2/control_model.pt` (StrokeNet, 27,194 params; 114,995 B fp32
  state dict, ~27,194 B int8 estimate) — the plain Arm 1 recipe trained
  only on UJI training-pool writers. No retraining, no fine-tuning, no
  handedness/Hershey data (C3's rejected lever stays rejected).
- **Routing rule (ONE rule, exact):** if the input carries a pen-point
  stream with at least one stroke containing at least 2 points, v8's
  prediction is the STROKE model's prediction (stroke-primary). Otherwise
  (image-only input, empty stream, or a degenerate single-point stream)
  v8's prediction is the KEEPER's prediction, produced by calling the
  keeper code path directly. No confidence blending, no trigger set, no
  third path.
- **Why stroke-primary and not a confusable-trigger cascade:** the stroke
  model is better than the keeper overall on this distribution and on
  every named problem pair, so consulting it only when the keeper's
  top-2 fall in a confusable set would discard most of its advantage and
  would inherit exactly the trigger-coverage failure that made v7
  Path A unsound (oracle gain +0.38 pts on the representative slice).
  The product demo natively captures real pen points, so stroke input
  is the normal case and image-only is the exception.
- **Plain claim statement (required for this routing):** when pen points
  are present, v8's exam claim is EXACTLY the stroke control model's
  claim — the hybrid adds nothing on those samples beyond choosing the
  better instrument. v8's additional claim is only the exact-fallback
  guarantee for image-only input (Gate 5). Gates 2 and 3 therefore
  reduce to equalities under this routing; the substantive gates are
  1 (headline vs both single models), 4 (size), and 5 (fallback).

## Fresh exam (frozen; built for this run, scored ONCE, all three systems in one sitting)

- Source: UJIpenchars2 (CC BY 4.0, as logged for the stroke experiment),
  ASCII letters only, case-merged, identical featurization / rendering /
  keeper preprocessing as the stroke experiment and C3.
- Partition: a NEW partition, distinct from both the stroke experiment's
  trn/tst split and C3's id%3 split. Exam writers = the writers that are
  in NEITHER component's actual UJI training pool: not among the stroke
  experiment's 40 `trn` writers (Arm 1's training pool, the recipe the
  control model follows) and not among C3 control's 40-writer training
  pool (id % 3 != 0). That intersection-complement is exactly 17 writers
  (ids 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57,
  60), 104 samples each = **1,768 exam samples**. (The task brief asked
  for 20 writers; only 17 exist outside both training pools, so the
  largest clean subset is used and n is reported honestly.)
- **Disclosure:** UJI has only 60 writers and every one of them has
  appeared in some earlier split of this project. These 17 writers were
  NEVER trained on by either component, but they were EVALUATED before:
  they are a subset of the stroke experiment's test writers and of C3's
  exam writers, so both stroke models and the keeper have published
  scores on supersets containing these samples. This exam is fresh as a
  partition and is writer-disjoint from all training; it is not virgin
  data for the project, and v8's numbers must be read with that caveat.
- Systems scored on the identical 1,768 samples in one sitting: keeper
  v2 (on renders), stroke control alone, v8 hybrid (stroke-primary).

## Keeper gates (ALL must pass or v8 is rejected; frozen)

1. Headline: v8 exam accuracy >= stroke control alone on this exam AND
   >= keeper v2 on this exam + 0.03.
2. a->q: v8 exam a->q rate <= stroke control's on this exam.
3. Pairs: v8 sum of combined pair errors (a/q + e/f + n/h) <= stroke
   control's sum on this exam.
4. Size: total deployed size <= 0.5 MB — keeper int8 (259,143 B) +
   stroke model (27,194 B int8 estimate; 114,995 B even at fp32) +
   routing code.
5. Fallback exactness: on image-only inputs (the exam renders with the
   stroke stream dropped), v8's output agrees with keeper v2's output
   on 100% of samples.

## Adversarial probes (named, reported with the results)

- **P-fallback:** the full exam re-run in image-only mode (stroke stream
  dropped) — the hybrid must degrade to exactly the keeper, never to a
  third behavior (this is also Gate 5, probed on every sample).
- **P-degenerate:** exam samples with the stroke stream replaced by a
  single point / emptied — routing must fall back to the keeper on
  100% of them, no crashes, no stroke-model garbage predictions.

## Terms / cost

UJIpenchars2 CC BY 4.0 and EMNIST (NIST SD19, public domain) only, as
previously logged. No frontier-model output in data, labels, filtering,
or selection. No UNIPEN. No recovered trajectories. Local CPU only,
**0 Colab units**. Nothing published or pushed; the demo is not modified.

## Results (v8, single exam evaluation, as pre-registered)

All three systems scored on the identical 1,768 exam samples (17 writers:
12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60) in
one sitting (`hybrid-v8/eval_v8.py` -> `results_v8.json`). Every exam
sample carried a real stroke stream, so routing sent 1,768 / 1,768 to
the stroke model and v8's stroke-present numbers are the stroke
control's, exactly as the pre-registration said they would be.

| Metric (exam) | Keeper v2 | Stroke control | v8 hybrid |
|---|---|---|---|
| Overall accuracy | 0.8292 | 0.8671 | **0.8671** |
| a->q | 0.0735 | 0.0000 | **0.0000** |
| a/q combined err | 0.2353 | 0.1250 | **0.1250** |
| e/f combined err | 0.2868 | 0.1176 | **0.1176** |
| n/h combined err | 0.2426 | 0.2500 | **0.2500** |
| Combined pair sum (a/q+e/f+n/h) | 0.7647 | 0.4926 | **0.4926** |
| Primary directed sum | 0.1176 | 0.0735 | **0.0735** |

Probes: **P-fallback** — image-only mode agreed with keeper v2 on
1,768 / 1,768 samples (100%). **P-degenerate** — empty and single-point
stroke streams routed to the keeper on 100% of samples (both probes
exercise the routing predicate directly; they verify the fallback
guarantee, not a learned behavior).

### Gates — 5 of 5, PASS

1. Headline (v8 >= stroke control AND >= keeper + 0.03): **PASS** —
   0.8671 >= 0.8671 (equality, per the stroke-primary claim) and
   0.8671 >= 0.8592 (+3.79 pts over the keeper's 0.8292).
2. a->q <= stroke control's: **PASS** — 0.0000 <= 0.0000.
3. Combined pair sum <= stroke control's: **PASS** — 0.4926 <= 0.4926.
4. Size <= 0.5 MB: **PASS** — keeper int8 259,143 B + stroke int8
   estimate 27,194 B + routing code 5,057 B = 291,394 B (even with the
   stroke model at fp32, 115,639 B, the total is 379,839 B).
5. Fallback exactness 100%: **PASS** — 1.000 agreement.

**Verdict: v8 is a KEEPER — all 5 gates pass.** Read it with the
pre-registration's caveats, which are part of the verdict: (1) on
stroke-present input v8 IS the stroke control model — the hybrid's own
contribution is choosing the better instrument plus the exact
image-only fallback, nothing more; (2) the exam writers were never
trained on by either component but were evaluated in earlier project
splits, so this is not a virgin-data result; (3) this keeper claim is
for the hybrid system on UJI-distribution pen input — v2 remains the
image-only keeper and its 95.24% EMNIST claim is untouched. The honest
product statement: when the demo has real pen points (its normal
case), the routed system scores 86.7% on unseen writers with a->q
eliminated, vs 82.9% for the image model alone on the same writers.

Cost: local CPU only, **0 Colab units**. Nothing published or pushed;
the demo was not modified. Artifacts in `hybrid-v8/`: eval_v8.py,
results_v8.json, pred_v8.npz, exam_samples_v8.pkl.

---

# v8 engine swap — H1 replaces the C3 control as the stroke engine (2026-11-19)

Rusty approved adopting the stroke-data-hunt H1 model as the v8
hybrid's stroke engine (adoption only — NOT publication; nothing
pushed, demo untouched). This note supersedes the v8 section above
ONLY in its Component 2; the G2 pre-registration, gates, and results
stand as the record of the original (control) configuration.

- **New Component 2:** H1-COMBINED-AUG-N12 (StrokeNet, 27,194 params),
  trained on the UJI C3-partition train pool + Character Trajectories
  (CC BY 4.0) + HWRT letters (ODbL 1.0), one frozen-augmentation copy
  per original; provenance in `stroke-exp/stroke-data-hunt/DESIGN.md`
  and `SOURCES.md`. Packaged engine file:
  `hybrid-v8/stroke_engine_h1_model.pt` (byte-identical copy of
  `stroke-exp/stroke-data-hunt/h1_combined_aug_n12_model.pt`, sha256
  9adeb60dffbb7c18778d3d61a52639f084f204026ea72ffc62721e879c16d2dc).
- **Only integration delta:** stroke featurization at **npts=12**
  (was 64). Routing rule unchanged (stroke-primary, exact keeper
  fallback); Component 1 (keeper v2 int8) byte-identical.
- **Parity (before scoring):** H1 through the hybrid stroke path
  reproduced its standalone C3-exam predictions exactly
  (2,080 / 2,080; features array-identical to `data_n12.npz`); the
  harness reproduced the recorded control predictions exactly
  (2,080 / 2,080); on the v8 exam the keeper, control, and
  hybrid-control lines reproduced `pred_v8.npz` exactly
  (1,768 / 1,768 each), so the image fallback is unchanged.
- **Re-score on the v8 exam (same 17 writers, 1,768 samples):**

  | Metric (exam) | Keeper v2 | v8 hybrid (control) | v8 hybrid (H1) |
  |---|---|---|---|
  | Overall accuracy | 0.8292 | 0.8671 | **0.9152** |
  | a->q | 0.0735 | 0.0000 | **0.0000** |
  | Combined pair sum (a/q+e/f+n/h) | 0.7647 | 0.4926 | **0.2721** |
  | Primary directed sum | 0.1176 | 0.0735 | **0.0588** |

  Every named pair improved individually (a/q 0.1250 -> 0.0882,
  e/f 0.1176 -> 0.0735, n/h 0.2500 -> 0.1103); P-fallback and
  P-degenerate probes 1.000 for the H1 configuration; deployed size
  296,637 B (int8 estimate) / 385,410 B (fp32 engine), within the
  0.5 MB gate.
- **Adoption rule (frozen for this step):** stick only if
  hybrid-with-H1 >= hybrid-with-control overall on the v8 exam with
  a->q at 0.0 and no problem-pair regression. **All conditions hold —
  the swap sticks. H1 is the v8 hybrid's stroke engine as of
  2026-11-19.** The earlier stroke-v2 L1-N8 adoption proposal is
  moot (never adopted; H1 supersedes it).
- Caveats unchanged from the v8 section: exam writers never trained
  on by either component but evaluated in earlier project splits
  (not virgin data); v2 remains the image-only keeper with its
  95.24% EMNIST claim untouched. Full record:
  `hybrid-v8/ADOPTION-H1.md`, `hybrid-v8/eval_v8_h1.py`,
  `hybrid-v8/results_v8_h1.json`, `hybrid-v8/pred_v8_h1.npz`.

Cost: local CPU only, **0 Colab units**. Nothing published or pushed;
the demo was not modified.
