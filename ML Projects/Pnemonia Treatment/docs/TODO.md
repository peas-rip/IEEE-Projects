# docs/TODO.md — updated after Phase 4

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

## Phase 1-4: full-stack web app

- [x] `scripts/dump_predictions.py` — per-member probability dump (`results/test_predictions.npz`)
- [x] `src/pneumonia/service.py` — `PipelineService`: load once, predict, reblend
- [x] `app/` — FastAPI backend (health, predict, reblend, testset/evaluate, testset/curves, results, samples, figures)
- [x] `frontend/` — Vite + vanilla JS SPA (Predict, Playground, Results, About)
- [x] `run_app.bat` / `run_app.sh` — one-command launcher
- [x] 30 curated sample images in `samples/`
- [x] `tests/test_service.py` (17 tests) + `tests/test_api.py` (17 tests)
- [x] `docs/APP.md` — web app setup guide
- [ ] README updated to app-first quickstart
- [ ] `docs/TODO.md` tick-off complete

## Still open, deliberately

- [ ] Per-image prediction CSV (no GPU, ~1 h) — would enable ROC curves,
      calibration and the threshold sweep in `docs/EXERCISES.md` Labs 5–6.
- [ ] Threshold selection on the 1,044-image validation split (no GPU).
- [ ] `CITATION.cff` still carries a placeholder author/repository — fill in
      real values before publishing.
- [ ] ONNX export (optional feature) — plan in `.opencode/plan/onnx-feature.md`
- [ ] Checkpoint distribution: ~1.05 GiB ships in the ZIP, not in git.
