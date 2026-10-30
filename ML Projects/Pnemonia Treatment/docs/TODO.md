# Packaging TODO — educational release

Goal: turn the working reproduction into a hand-over package for engineering
students. No further training; models stay exactly as evaluated in
`results/reproduction_run.json`.

- [x] Persist results: `--json` flag on the three evaluation scripts
- [x] Record the completed Colab run in `results/reproduction_run.json`
- [x] `scripts/make_figures.py` + `docs/figures/` (confusion matrices, comparison chart)
- [x] `docs/RESULTS.md` — all tables, gap analysis, threats to validity
- [x] `docs/SETUP.md` — Colab/Drive layout, known-good numbers, laptop path
- [x] `docs/EXERCISES.md` — labs that need no retraining
- [x] `docs/REPORT_TEMPLATE.md` — IEEE-style scaffold pre-filled with real numbers
- [x] `LICENSE`, `NOTICE`, `CITATION.cff`
- [x] README rewrite around the real workflow
- [x] Tests green, commit on `colab-full-training`

## Done in the same pass (found while packaging)

- [x] Single checkpoint resolver (`src/pneumonia/checkpoints.py`) so the three
      evaluation scripts stop disagreeing about where Stage 2 PneumoNet-v1
      lives; `PNEUMONIA_CHECKPOINT_DIR` overrides the root for Colab/Drive.
- [x] JSON payloads wrote checkpoint *paths* under a `"weights"` key — renamed to
      `"checkpoints"`.
- [x] `pneumonia infer` hard-coded PneumoNet-v1 and would have failed on 6 of the
      7 checkpoints; it now takes `--config`, `--model`, `--image-size` and
      `--labels`.

## Still open, deliberately

- [ ] Per-image prediction CSV (no GPU, ~1 h) — would enable ROC curves,
      calibration and the threshold sweep in `docs/EXERCISES.md` Labs 5–6.
- [ ] Threshold selection on the 1,044-image validation split (no GPU).
- [ ] `CITATION.cff` still carries a placeholder author/repository — fill in
      real values before publishing.
- [ ] Checkpoint distribution: ~700 MB is not in the repo; needs a host and a
      documented link in `docs/SETUP.md`.
