# Task Tracker (TASKS.md)

Status legend: TODO | IN PROGRESS | DONE | BLOCKED | NOT RUN

## Phase 0 — Environment & Dependencies
- [x] T01 Inspect workspace, locate paper, render/read all pages (DONE)
  - Objective: Confirm paper presence and extract content.
  - Prerequisites: PyMuPDF installed.
  - Expected files: research/extracted-paper.md, research/pages/page_*.png.
  - Key behavior: All 16 pages read; paper metadata recorded.
  - Test/verification: ls research/pages shows 16 images.
  - Acceptance: Paper title, authors, venue, DOI recorded.
- [x] T02 Install ML dependencies (DONE)
  - Objective: Install torch, torchvision, scikit-learn, Pillow, matplotlib, tqdm.
  - Prerequisites: Python 3.11, pip.
  - Expected files: none (site-packages).
  - Key behavior: `import torch, torchvision, sklearn, PIL` succeed.
  - Test/verification: python -c "import torch; print(torch.__version__)".
  - Acceptance: All imports succeed; CUDA availability reported.
- [x] T03 Create dependency files (requirements.txt, pyproject.toml)
  - Objective: Declare dependencies with justified version constraints.
  - Prerequisites: T02.
  - Expected files: requirements.txt, pyproject.toml.
  - Key behavior: pip install -r requirements.txt reproduces env.
  - Test/verification: pip check.
  - Acceptance: Dependency file committed.

## Phase 1 — Data Contracts & Configuration
- [x] T04 Create configs/default.yaml
  - Objective: Central config for hyperparameters and paths.
  - Prerequisites: T03.
  - Expected files: configs/default.yaml.
  - Key behavior: lr=1e-4, batch_size=32, epochs=10, image_size=224, ensemble weights placeholders.
  - Test/verification: python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))".
  - Acceptance: Config parses and contains required keys.
- [x] T05 Implement data module (src/pneumonia/data.py)
  - Objective: Dataset loading, transforms (resize 224x224, normalize, augment), 80:20 split, synthetic fixture.
  - Prerequisites: T04.
  - Expected files: src/pneumonia/data.py.
  - Key behavior: make_synthetic_fixture() generates N grayscale 224x224 images with labels; DataLoader yields (batch, 3, 224, 224) float tensors.
  - Test/verification: pytest tests/test_data.py.
  - Acceptance: Synthetic fixture shapes and dtypes correct; split stratified.

## Phase 2 — Core Model Components
- [x] T06 Implement PneumoNet-v1 (src/pneumonia/models.py)
  - Objective: Custom CNN matching paper architecture.
  - Prerequisites: T05.
  - Expected files: src/pneumonia/models.py.
  - Key behavior: input (N,3,224,224) -> output (N,2); layer filter counts match paper (128/256/512 etc.).
  - Test/verification: pytest tests/test_models.py (forward shape).
  - Acceptance: Output shape correct; parameter count > 0.
- [x] T07 Implement PneumoNet-v2 (src/pneumonia/models.py)
  - Objective: Custom CNN matching paper architecture (lower LR variant).
  - Prerequisites: T06.
  - Expected files: src/pneumonia/models.py (extended).
  - Key behavior: input (N,3,224,224) -> output (N,2).
  - Test/verification: pytest tests/test_models.py.
  - Acceptance: Forward pass runs without error.
- [x] T08 Implement transfer-learning wrappers (src/pneumonia/transfer.py)
  - Objective: Wrap VGG-16, Inception-v3, Xception, MobileNet-v2 from torchvision with new heads.
  - Prerequisites: T07.
  - Expected files: src/pneumonia/transfer.py.
  - Key behavior: each wrapper accepts (N,3,224,224) -> (N,2); pretrained ImageNet weights.
  - Test/verification: pytest tests/test_models.py.
  - Acceptance: All four wrappers forward-pass correctly.

## Phase 3 — Loss, Optimizer, Training
- [x] T09 Implement training loop (src/pneumonia/train.py)
  - Objective: SGD lr=1e-4, Categorical Crossentropy, epoch loop, checkpoint save/load.
  - Prerequisites: T08.
  - Expected files: src/pneumonia/train.py.
  - Key behavior: train_model(model, train_loader, val_loader, epochs) returns history; save/load checkpoint works.
  - Test/verification: pytest tests/test_train_eval.py.
  - Acceptance: Single epoch on synthetic data completes; loss decreases (not NaN).
- [x] T10 Implement evaluation module (src/pneumonia/evaluate.py)
  - Objective: accuracy, precision, recall, F1, AUC, AP, ROC, confusion matrix.
  - Prerequisites: T09.
  - Expected files: src/pneumonia/evaluate.py.
  - Key behavior: compute_metrics(y_true, y_pred, y_prob) returns dict; plot functions save figures.
  - Test/verification: pytest tests/test_train_eval.py.
  - Acceptance: Metrics match sklearn on hand-calculated tiny example.

## Phase 4 — Ensemble Pipeline
- [x] T11 Implement weighted ensemble (src/pneumonia/ensemble.py)
  - Objective: DetectionEnsemble (4 models) and ClassificationEnsemble (3 models) with weighted average.
  - Prerequisites: T06, T07, T08.
  - Expected files: src/pneumonia/ensemble.py.
  - Key behavior: forward pass returns weighted probabilities and final label; two-stage inference: if Normal stop, else classify bacterial/viral.
  - Test/verification: pytest tests/test_ensemble.py.
  - Acceptance: Two-stage inference on synthetic fixture returns valid label.

## Phase 5 — CLI & Packaging
- [x] T12 Implement CLI entry point (src/pneumonia/cli.py)
  - Objective: `python -m pneumonia train|evaluate|infer` commands.
  - Prerequisites: T10, T11.
  - Expected files: src/pneumonia/cli.py, scripts/train.py.
  - Key behavior: train on synthetic data, evaluate, infer on a single image.
  - Test/verification: run CLI commands manually.
  - Acceptance: All three subcommands run end-to-end on synthetic data.
- [x] T13 Create pyproject.toml / package setup
  - Objective: Installable package with console script entry point.
  - Prerequisites: T12.
  - Expected files: pyproject.toml.
  - Test/verification: pip install -e . ; pneumonia --help.
  - Acceptance: Package installs and console script works.

## Phase 6 — Testing & Verification
- [x] T14 Write unit/integration tests (tests/)
  - Objective: Cover models, data, ensemble, train/eval.
  - Prerequisites: T05–T11.
  - Expected files: tests/test_*.py.
  - Key behavior: pytest passes.
  - Test/verification: pytest -q.
  - Acceptance: All tests PASS.
- [x] T15 Run end-to-end smoke test on synthetic data
  - Objective: Verify full train -> evaluate -> infer pipeline.
  - Prerequisites: T14.
  - Expected files: results/smoke_test_*.json (if metrics logged).
  - Test/verification: python scripts/train.py --config configs/default.yaml --data synthetic.
  - Acceptance: Pipeline completes; metrics logged; no errors.

## Phase 7 — Documentation
- [x] T16 Write README.md with quickstart, prerequisites, installation, examples, troubleshooting, limitations
  - Objective: Usable project documentation.
  - Prerequisites: T15.
  - Expected files: README.md.
  - Key behavior: Quickstart commands are ones actually verified.
  - Test/verification: Run README quickstart commands.
  - Acceptance: README commands verified; limitations section present.
- [x] T17 Final review: git diff, no secrets/unrelated files, acceptance criteria vs. paper
  - Objective: Clean handoff.
  - Prerequisites: T16.
  - Expected files: git status clean of unrelated files.
  - Test/verification: git diff --stat; review against acceptance criteria.
  - Acceptance: All tasks marked DONE/BLOCKED/NOT RUN honestly; discrepancies documented.
