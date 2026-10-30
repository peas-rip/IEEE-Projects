# Colab Run — Verified Results

Curated extract of the end-to-end run on Google Colab (Tesla T4, Python 3.13,
torch 2.11.0+cu130). This is the **authoritative source** for every figure in
`results/reproduction_run.json`. The full notebook output is not committed;
this file keeps only the evaluation blocks and the environment header.

## Environment

| Item | Value |
|---|---|
| Runtime | Google Colab, Tesla T4 (15.6 GB VRAM) |
| Python | 3.13 |
| torch | 2.11.0+cu130 |
| torchvision | (default Colab) |
| Dataset | Kermany, prepared on Drive: 5,216 train / 624 test (Stage 1), 3,875 train / 390 test (Stage 2) |
| Input | 224×224 JPEG cache, quality 92, built once on local disk |
| Training | 15 epochs, SGD lr 1e-4, AMP on, batch 32 (Stage 1 custom/VGG) / 16 (Inception-v3 and Stage 2) |
| Seeds | 42 |

## Stage 1 — per-model (Normal vs Pneumonia, 624 images)

| Model | Accuracy | AUC | Confusion matrix |
|---|---:|---:|---|
| PneumoNet-v1 | 0.8285 | 0.9456 | `[[134, 100], [7, 383]]` |
| PneumoNet-v2 | 0.8510 | 0.9298 | `[[164, 70], [23, 367]]` |
| VGG-16 | 0.9119 | 0.9690 | `[[193, 41], [14, 376]]` |
| Inception-v3 | 0.6458 | 0.7969 | `[[13, 221], [0, 390]]` |

Inception-v3 collapsed: NORMAL recall 0.056 (13 / 234), PNEUMONIA recall 1.000.

## Stage 1 — ensemble (paper weights 5/4/3/1)

| Metric | Value |
|---|---:|
| Accuracy | **0.8606** |
| AUC | **0.9634** |
| AP | 0.9616 |
| Precision (pneumonia) | 0.8244 |
| Recall (pneumonia) | 0.9872 |
| F1 | 0.8985 |
| Confusion matrix | `[[152, 82], [5, 385]]` |

Grid search (1–6) found weights 1/1/6/1 → 0.8990, but those are
test-fitted and not the reported number.

## Stage 2 — per-model (Bacterial vs Viral, 390 pneumonia images)

| Model | Accuracy | AUC | Confusion matrix |
|---|---:|---:|---|
| PneumoNet-v1 | 0.8205 | 0.9396 | `[[181, 61], [9, 139]]` |
| MobileNet-v2 | **0.8513** | 0.9370 | `[[197, 45], [13, 135]]` |
| Inception-v3 (Xception substitute) | 0.7026 | 0.7845 | `[[228, 14], [102, 46]]` |

## Stage 2 — ensemble (grid-searched weights 2/3/5)

| Metric | Value |
|---|---:|
| Accuracy | **0.8949** |
| AUC | **0.9390** |
| AP | 0.9395 |
| Confusion matrix | `[[218, 24], [17, 131]]` |

## Full two-stage pipeline (3-class, 624 images)

| Metric | Value |
|---|---:|
| Overall accuracy | **0.7853** (490 / 624) |
| AUC (OvR) | **0.9503** |
| AP (macro) | 0.9135 |
| Confusion matrix | `[[152, 6, 76], [3, 207, 32], [2, 15, 131]]` |

### Stage 1 breakdown (Normal vs Pneumonia)

| Metric | Value |
|---|---:|
| Accuracy | 0.8606 |
| Precision / Recall / F1 (pneumonia) | 0.8244 / 0.9872 / 0.8985 |
| Confusion matrix | `[[152, 82], [5, 385]]` |
| Routed to Stage 2 | 385 |

### Stage 2 breakdown (Bacterial vs Viral, routed set)

| Metric | BACTERIAL | VIRAL |
|---|---:|---:|
| Precision | 0.9324 | 0.8037 |
| Recall | 0.8661 | 0.8973 |
| F1 | 0.8980 | 0.8479 |
| Accuracy (routed) | 0.8779 (338 / 385) | — |
| Accuracy (misses counted wrong) | 0.8667 (338 / 390) | — |
| Confusion matrix | `[[207, 32], [15, 131]]` | — |

## Ablations (no retraining)

| Configuration | Accuracy | Δ vs baseline | Images fixed |
|---|---:|---:|---:|
| Baseline (paper weights) | 0.7853 | — | — |
| Drop Inception-v3 from Stage 1 | **0.8301** | +4.48 | 28 |
| Stage 2 weights 2/3/5 | 0.7981 | +1.28 | 8 |
| Both | 0.8429 | +5.76 | 36 |

## Error decomposition

| Cause | Images | Share |
|---|---:|---:|
| Healthy lungs flagged as pneumonia | 82 | 61.2% |
| Pneumonia missed by Stage 1 | 5 | 3.7% |
| Subtype confusion on correctly detected pneumonia | 47 | 35.1% |
| **Total** | **134** | 100% |

## Reproducing this run

All numbers are in `results/reproduction_run.json`. To re-run on Colab:

```bash
python scripts/evaluate_stage1_ensemble.py --json results/stage1_ensemble.json
python scripts/evaluate_stage2_ensemble.py --json results/stage2_ensemble.json
python scripts/evaluate_full_pipeline.py --ablate --json results/pipeline_ablation.json
python scripts/make_figures.py
```

Expected values to check against are in `docs/SETUP.md` §6.
