# ADOPTION — H1 as the v8 hybrid's stroke engine (2026-11-19)

Status: **ADOPTED** (Rusty approved the adoption; NOT publication —
nothing pushed, the demo is untouched). Executed and recorded
2026-11-19, local CPU, 0 Colab units.

## The swap

- **Out:** frozen C3 CONTROL (`write-it-right/stroke-v2/control_model.pt`,
  featurized at N = 64) — retained untouched as the historical engine.
- **In:** H1-COMBINED-AUG-N12 (`stroke-exp/stroke-data-hunt/`
  `h1_combined_aug_n12_model.pt`; packaged copy in this folder as
  `stroke_engine_h1_model.pt`, sha256
  9adeb60dffbb7c18778d3d61a52639f084f204026ea72ffc62721e879c16d2dc,
  byte-identical to the source artifact).
- **Integration delta (only):** stroke inputs are featurized with
  `stroke-exp/common.py featurize(strokes, npts=12)` instead of
  `npts=64`. Same 5 features (x, y, dx, dy, pen_lift), same StrokeNet
  architecture (27,194 params), same ONE routing rule (stroke-primary
  when a real pen stream is present, exact keeper fallback otherwise).
- **Unchanged:** Component 1 is still the frozen keeper v2
  (`write_it_right_v2_int8.onnx`, 259,143 B), byte-identical, and the
  fallback path reproduces its predictions exactly (see parity P0d and
  the P-fallback probe below).

H1's provenance (data-hunt DESIGN.md / SOURCES.md): trained on UJI
C3-partition train pool (3,536) + Character Trajectories (UCI 175,
CC BY 4.0; 2,858) + HWRT letters (Zenodo 50022, ODbL 1.0; 3,638),
each original plus one frozen-augmentation copy (20,064 rows),
`train_stroke.py` unmodified, seed 0. Augmentation was training-time
only; inference is the plain 12-point featurization above.

## Parity checks (all run BEFORE the v8-exam scoring)

| Check | Result |
|---|---|
| P0a — featurize(C3 exam raw, N=12) vs `data_n12.npz` X_test | array-identical |
| P0b — H1 through the hybrid stroke path vs H1 standalone preds (C3 exam) | **2,080 / 2,080 exact** |
| P0c — control through the same harness vs recorded control preds (C3 exam) | 2,080 / 2,080 exact |
| P0d — v8 exam reproduction: keeper / control / hybrid-control vs `pred_v8.npz` | **1,768 / 1,768 each, exact** |
| Image fallback (image-only inputs) reproduces keeper v2 | 100% (P0d + P-fallback probe) |

## Re-score on v8's OWN exam (G2 partition: 17 writers, 1,768 samples, identical items)

All 1,768 samples carried real stroke streams and routed to the
stroke engine, so each hybrid's numbers are its engine's, exactly as
the v8 pre-registration describes for stroke-primary routing.

| Metric (exam) | Keeper v2 | Hybrid (control engine) | Hybrid (H1 engine) |
|---|---|---|---|
| Overall accuracy | 0.8292 | 0.8671 (1,533) | **0.9152 (1,618)** |
| a→q | 0.0735 | 0.0000 | **0.0000** |
| a/q combined err | 0.2353 | 0.1250 | **0.0882** |
| e/f combined err | 0.2868 | 0.1176 | **0.0735** |
| n/h combined err | 0.2426 | 0.2500 | **0.1103** |
| Combined pair sum | 0.7647 | 0.4926 | **0.2721** |
| Primary directed sum | 0.1176 | 0.0735 | **0.0588** |

Probes for the H1 hybrid: **P-fallback** agreement 1.000;
**P-degenerate** (empty / single-point streams route to the keeper)
1.000. H1 hybrid's weakest letters on this exam: k 0.7941, i 0.8235,
p 0.8529 (control engine's floor was u 0.6912). Size: keeper int8
259,143 B + H1 int8 estimate 27,194 B + routing code 10,300 B =
296,637 B (385,410 B with H1 at fp32) — within the 0.5 MB gate.

## Adoption rule and verdict

Frozen rule for this step: the swap sticks ONLY if hybrid-with-H1
overall ≥ hybrid-with-control overall on the v8 exam, a→q stays 0.0
on the stroke path, and no problem-pair regression. **All hold** —
overall +4.81 pts (0.9152 vs 0.8671), a→q 0.0000, and every named
pair improved individually (a/q, e/f, and n/h combined errors all
fell; the sum fell 0.4926 → 0.2721). **Adoption sticks: H1 is now
the v8 hybrid's stroke engine.** With H1 substituted as the stroke
component, the re-scored hybrid also satisfies the original five G2
gates on this exam (headline, a→q, pair sum vs the engine alone —
an equality under stroke-primary routing — size, fallback exactness).

Caveats, carried over from the v8 pre-registration and still in
force: the exam writers were never trained on by either component
(H1's non-UJI corpora are provenance-disjoint from UJI), but they
were evaluated in earlier project splits, so this is not a
virgin-data result. v2 remains the image-only keeper; its 95.24%
EMNIST claim is untouched. The honest product statement updates to:
when the demo has real pen points (its normal case), the routed
system scores **91.5%** on unseen writers with a→q eliminated, vs
82.9% for the image model alone on the same writers.

Artifacts in this folder: `eval_v8_h1.py` (routing + evaluation code
of record for the H1 configuration), `results_v8_h1.json`,
`pred_v8_h1.npz`, `stroke_engine_h1_model.pt`. The original G2
artifacts (`eval_v8.py`, `results_v8.json`, `pred_v8.npz`) are
unchanged and remain the record of the control configuration.
