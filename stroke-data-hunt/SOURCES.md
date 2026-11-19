# Stroke-data hunt — source survey (2026-11-19)

Goal: find public **online** (pen-point stroke) corpora of isolated Latin
letters to break the Write It Right stroke model's data wall (3,536 UJI
train letters / 40 writers on the C3 partition; ladder plateaued at
89.4%, diagnosis: training-data volume and writer diversity).

Project license standard (from write-it-right/PROVENANCE.md): license
verified **at source** and recorded before ingest. Sources whose license
is NonCommercial / research-only / encumbered, or whose license cannot
be verified at source, are **reported, not ingested** — the same rule
that kept UNIPEN out of C3.

## Survey table

| Source | Isolated Latin letters (case-merged a–z) | Writers | Format | License at source | Verdict |
|---|---|---|---|---|---|
| UJIpenchars2 (already in use) | 7,920 (60 writers × 132) | 60 | UNIPEN-like text | CC BY 4.0 (UCI 177) | in use |
| **Character Trajectories (UCI 175)** | **2,858** (20 letters: a b c d e g h l m n o p q r s u v w y z) | **1** | (x,y,force) velocity trajectories, .mat | **CC BY 4.0** (UCI dataset page) | **INGESTED** |
| **HWRT database (Zenodo 10.5281/zenodo.50022)** | **3,638** (all 26 letters; lowercase 't' absent as a symbol — its 56 samples are uppercase T) | **192** (distinct user_ids) | CSV, JSON stroke lists | **ODbL 1.0** (Zenodo record metadata: `odc-odbl`) | **INGESTED** (see ODbL note) |
| UNIPEN train_r01_v07 (iUF) | 37,506 isolated lowercase (devset table; trainset similar scale) + 16,414 uppercase | ~200+ | UNIPEN format | iUF terms (in-distribution README): "PERMISSION IS HEREBY GRANTED TO USE THE DATA FOR RESEARCH PURPOSES. IT IS NOT ALLOWED TO DISTRIBUTE THIS DATA FOR COMMERCIAL PURPOSES." Conditions of Use §II.b do allow training commercial recognizers, but with obligations (logo in product docs, publication policy, no redistribution); full trainset is sold by iUF; third-party mirrors are of doubtful channel legitimacy | **NOT ingested** — encumbered terms + standing project decision (C3, 2026-11-19). Revisit only as an explicit decision to obtain it from iUF under its conditions |
| CROHME (TC11 / isical) | symbols extractable from ~10k expressions (symbol-level GT in InkML) | hundreds | InkML | **CC BY-NC-SA 3.0** ("freely available only for research purpose without any commercial use" — isical CROHME page) | NOT ingested — NonCommercial |
| MathWriting (Google) | letters extractable from 230k human-written expressions (no symbol-level GT published for letters in expressions; would need segmentation) | many (Google contributors) | InkML | **CC BY-NC-SA 4.0** (repo README + paper) | NOT ingested — NonCommercial |
| OnHW-chars (Fraunhofer IIS) | 31,275 (52 classes) — **but IMU sensor streams, no pen-tip x/y at all** ("nothing records where the ink goes") | 119 | CSV sensor data | **CC BY-NC 4.0** (HF mirror README, Fraunhofer terms) | NOT ingested — NonCommercial, and not stroke-coordinate data |
| IRONOFF (IRESTE/LORIA) | ~32,000 isolated characters claimed | ~373–700 | UNIPEN format | No public license/download found; per Llorens et al., "the IRONOFF database can be bought from the University of Nantes" | NOT ingested — no verifiable license at a public source |
| IAM-OnDB (Univ. Bern) | none — words/lines/pages only, no isolated-letter GT | 400+ | stroke XML | Distribution only after signing the IAM license agreement (research use) | Skipped — agreement-gated, and letter segmentation from words would be unprincipled (per task rule) |
| Pen-Digits (UCI 81) | 0 — digits only | 44 | UNIPEN-ish | CC BY 4.0 (UCI) | Out of scope (no letters) |
| UJI Pen Characters v1 (UCI 177 subset) | subset of UJIpenchars2 | 11 | text | CC BY 4.0 | Already contained in v2 |
| Detexify raw dumps (GitHub mirrors) | large, but no license statement for the sample data at the project source | many | JSON | Unverifiable at source | NOT ingested — superseded by HWRT, which republishes ~90% of the same recordings under ODbL (see below) |

## Per-source license records (ingested sources)

### Character Trajectories — UCI dataset 175
- Source: https://archive.ics.uci.edu/dataset/175/character+trajectories
  (file: `character+trajectories.zip` → `mixoutALL_shifted.mat`,
  `trajectories.names`). Dataset DOI 10.24432/C58G7V (as listed by UCI).
- License as stated at source: the UCI dataset page carries
  "Creative Commons Attribution 4.0 International" (CC BY 4.0).
- Creator: Ben H. Williams, School of Informatics, University of
  Edinburgh (PhD study on primitive extraction).
- Content: 2,858 samples, **all from one writer**, WACOM tablet at
  200 Hz. Only characters with a **single pen-down segment** were kept.
  20 letters (no f, i, j, k, t, x — multi-stroke/dotted letters are
  excluded by construction). Per-letter counts 119–186.
- The stored channels are pen-tip **velocity** (x-vel, y-vel) + force,
  numerically differentiated and Gaussian-smoothed (σ=2), then
  normalized. Conversion to positions: cumulative integration of the
  velocity channels (constant dt ⇒ scale-free after the project's
  unit-box normalization), y-axis flipped to the UJI y-down convention.
  Orientation verified visually: integrated+flipped samples render as
  upright, legible letters (a e n h q g spot-checked).
- Style caveat: this writer's letters are **cursive single-stroke
  allographs** (e.g. cursive ɑ-form 'a'), quite unlike UJI's printed
  forms. It adds volume and one very different writer — not 40 writers
  of diversity.
- Converted: `converted/char_traj_samples.pkl` (2,858 samples, project
  sample-dict format) by `convert_char_traj.py`.
- Contamination: provenance is disjoint from UJI (different device,
  study, writer); no UJI exam-writer data can be present. No dedupe
  against the exam was needed beyond this provenance check.

### HWRT database of handwritten symbols — Zenodo 10.5281/zenodo.50022
- Source: https://zenodo.org/records/50022 (file `2015-01-28-data.tar`,
  140,790,596 bytes). Author: Martin Thoma (write-math.com); ~90% of
  the recordings were collected by Daniel Kirsch's Detexify project
  and republished here by the database author.
- License as stated at source: Zenodo record metadata license =
  **Open Data Commons Open Database License 1.0 (ODbL 1.0)**
  (`odc-odbl`); the DataCite record (10.5281/zenodo.50022) states the
  same rights.
- **ODbL note (why this is ingestible under the project standard):**
  ODbL is an open license with attribution + share-alike on the
  *database*. It obliges: attribution (given here and to be carried
  into PROVENANCE), and share-alike if a *derivative database* is
  publicly used/redistributed. This project does **not** redistribute
  the database or the converted derivative; conversion is internal.
  Model weights trained on the data are a "produced work" under ODbL
  §4.4, not a derivative database; the attribution notice travels with
  the model record. If the converted data or a fine-tuned dataset were
  ever published, ODbL share-alike + attribution would apply to it.
- Content: crowdsourced online recordings of LaTeX symbols from
  Detexify + write-math.com, each with a `user_id`. Letters are the
  51 single-letter symbols (A–Z, a–z; lowercase 't' does not exist
  as a symbol). Conversion (`convert_hwrt.py`): both the author's
  train and test CSVs used (our exam is UJI's; HWRT's own split is
  irrelevant here), geometry-only filters (no strokes / <2 points /
  single repeated point) — **3,638 samples kept, 0 dropped, from 192
  distinct user_ids**, per-letter 56–245. Strokes are JSON point
  lists in web-canvas coordinates (y down — same convention as UJI;
  verified by rendering spot checks: a e n h q t upright).
  Crowdsourced labels carry some noise (Detexify-era recordings are
  known to include misdraws/mislabels); no label curation was applied.
- Contamination: crowdsourced web data, disjoint provenance from UJI;
  no UJI exam data can be present.

## Bottom line of the hunt

The big online corpora are almost all NonCommercial (CROHME,
MathWriting, OnHW) or encumbered (UNIPEN). Under the project's
verified-license standard, the actually-ingestible new letters are:

- Character Trajectories: 2,858 letters, 1 writer, 20 letters.
- HWRT: 3,638 letters, 192 writers, all 26 letters.
- Combined with UJI train: 10,032 originals (2.84× the old pool).

Everything else is documented above for a future explicit licensing
decision (UNIPEN §II.b is the one worth a human decision: its terms do
permit training commercial recognizers, under obligations).
