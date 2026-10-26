# Pneumonia Detection & Classification (Paediatric Chest X-Ray)

A clean, reproducible reimplementation of the method from:

> **Enhancing Paediatric Pneumonia Detection and Classification Using Customized CNNs and Transfer Learning Based Ensemble Models**
> Shubham Godbole et al., *Int. Res. J. Multidisc. Technovation*, 6(6) (2024) 38–53.
> DOI: https://doi.org/10.54392/irjtmt2463 | License: CC BY 4.0

## What this project implements

Two weighted ensemble models for chest X-ray (CXR) images, exactly as described in the paper:

| Stage | Task | Base models | Paper-reported test accuracy |
|-------|------|-------------|------------------------------|
| 1 | Normal vs. Pneumonia (detection) | VGG-16, Inception-v3, PneumoNet-v1, PneumoNet-v2 | 98.03% |
| 2 | Bacterial vs. Viral (classification) | Xception, MobileNet-v2, PneumoNet-v1 | 91.93% |

**Inference flow:** input CXR → Stage 1 detection → if *Normal*, stop → if *Pneumonia*, Stage 2 classifies as *Bacterial* or *Viral*.

**Custom CNNs:** PneumoNet-v1 and PneumoNet-v2 (implemented from the paper's architecture descriptions in Sections 3.3–3.4).

**Transfer-learning baselines:** VGG-16, VGG-19, Inception-v3, ResNet-50, ResNet-101, DenseNet-121, DenseNet-169, MobileNet-v2 (Xception substituted with Inception-v3 backbone — see notes below).

## Repository layout

```
research/            paper-notes.md, method-spec.yaml, assumptions-and-open-questions.md,
                     implementation-plan.md, TASKS.md
src/pneumonia/       package: data, models, transfer, ensemble, train, evaluate, cli
tests/               unit + integration tests
configs/default.yaml hyperparameters (lr=1e-4 SGD, 80:20 split, 224x224)
scripts/train.py     thin CLI wrapper
results/             experiment outputs (created when you run experiments)
Paper/               original paper PDF (scanned; extracted under research/pages/)
```

## Prerequisites

- Python >= 3.9
- Pip
- CPU works fine; a GPU (>= 4 GB VRAM) speeds up training on the real dataset.

## Installation

```bash
# clone / open this folder
cd "D:\Projects\IEEE\IEEE-Projects\ML Projects\Pnemonia Treatment"

# create & activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell

# install dependencies + the package in editable mode
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quickstart (verified)

All commands below were run and verified on this machine (CPU-only PyTorch).

### 1) Train on a synthetic smoke-test fixture

The real Kermany dataset is gated behind an author request (see the paper's Data Availability section), so a synthetic fixture is used for smoke tests.

```bash
python -m pneumonia train --config configs/default.yaml --data synthetic
```

This trains `PneumoNet-v1` (SGD, lr=1e-4, Categorical Crossentropy) for the number of epochs in `configs/default.yaml` (default 10), saves checkpoints under `results/checkpoints/`, and writes `results/checkpoints/final_model.pt`.

### 2) Evaluate

```bash
python -m pneumonia evaluate --config configs/default.yaml --data synthetic \
  --model-weights results/checkpoints/final_model.pt
```

Prints accuracy, per-class precision/recall/F1, macro averages, confusion matrix, AUC, and Average Precision.

### 3) Infer on a single chest X-ray

```bash
python -m pneumonia infer --image path/to/your_xray.png \
  --checkpoint results/checkpoints/final_model.pt
```

Prints the predicted class (`Normal` / `Pneumonia`) and class probabilities.

### 4) Run the test suite

```bash
python -m pytest tests/ -q
```

## Configuration

Edit `configs/default.yaml` to change:
- `training.model`: `pneumonet_v1` | `pneumonet_v2` | `vgg16` | `vgg19` | `inception_v3` | `resnet50` | `resnet101` | `densenet121` | `densenet169` | `mobilenet_v2`
- `training.learning_rate`: paper-stated `0.0001` (SGD)
- `training.epochs`, `data.batch_size`, `data.image_size` (paper: 224)
- `data.train_ratio`: paper-stated `0.8`

## Using the real dataset

The paper's dataset (Kermany / Guangzhou Women and Children's Medical Center, 5,856 paediatric CXRs) is gated behind an author request in the paper's Data Availability section — but the same underlying dataset is publicly available on Kaggle as **"Chest X-Ray Images (Pneumonia)"** (`paultimothymooney/chest-xray-pneumonia`), which is what this project trains on.

Class distributions (Kaggle copy):

| Split | NORMAL | PNEUMONIA | PNEUMONIA breakdown |
|-------|--------|-----------|---------------------|
| train | 1,341 | 3,875 | bacterial 2,530 / viral 1,345 |
| test  | 234 | 390 | bacterial 242 / viral 148 |

The train split is class-imbalanced (~2.9:1 Pneumonia:NORMAL), so Stage 1 configs enable **class weighting** (`training.use_class_weights: true`) — without it the model collapsed to predicting every image as Pneumonia.

### Colab training

`colab_stage1_training.ipynb` trains the full 4-model Stage 1 detection ensemble on Google Colab (T4 GPU):

1. Runtime → Change runtime type → **T4 GPU**
2. Upload the notebook to Colab and run cells in order: mount Drive → clone this repo → install deps → authenticate Kaggle (`kaggle.json`) → fetch dataset via `kagglehub` → reorganize into Stage 1/Stage 2 layouts on Drive
3. Sanity-check run on a tiny subset, then train the 4 models sequentially (PneumoNet-v1, PneumoNet-v2, VGG-16, Inception-v3) with per-epoch checkpoints saved to Google Drive
4. Evaluate the detection ensemble with the paper's published weights (VGG-16:5, Inception-v3:4, PneumoNet-v2:3, PneumoNet-v1:1) and compare against the 98.03% / AUC 0.97 target

```bash
# or train any single model locally
python -m pneumonia train --config configs/stage1_vgg16.yaml --data /path/to/stage1/train
python -m pneumonia evaluate --config configs/stage1_vgg16.yaml --data /path/to/stage1/test --checkpoint results/stage1_vgg16/final_model.pt
```

## Notes & known deviations from the paper (INFERRED / DEFAULT)

1. **Ensemble aggregation.** The paper describes weighting and combining base-model probabilities (grid/genetic search over weights) but does not publish an explicit aggregation formula or the full set of optimal weights. This project uses a **weighted average of softmax probabilities** (engineering default). Published weights for the best detection combo are honored where available; classification-ensemble weights default to uniform and should be tuned on validation data.

2. **Xception.** torchvision does not provide Xception; this project substitutes an Inception-v3 backbone with a new head (documented in `src/pneumonia/transfer.py`).

3. **Input channels.** Chest X-rays are grayscale; the paper specifies a 3-channel 224×224 input. This project replicates the single channel to 3 channels (engineering default).

4. **Batch size / epochs / seeds.** Not stated in the paper. Defaults: batch_size=32, epochs=10, seed=42 (configurable).

5. **Adaptive pooling.** The paper's conv-trunk description can over-reduce small inputs; an `AdaptiveAvgPool2d((7,7))` is applied before the classifier to keep the dense input fixed (engineering default). The paper-stated input size is 224×224.

## Reproducibility

- Seeds: Python/numpy/torch seeded to 42 by default.
- Versions recorded at install: `torch 2.14.1`, `torchvision 0.29.1`, `scikit-learn 1.9.1`, `Pillow 13.0.0`, `numpy 2.4.6`, `matplotlib 3.11.2`.
- Training runs log per-epoch train/val loss and accuracy; checkpoints store model + optimizer state and epoch.

## Paper-reported results (reference only; not reproduced here)

These are the values stated in the paper. They could not be reproduced exactly because the real dataset and exact ensemble weights are not publicly available.

- Detection ensemble (Normal vs. Pneumonia): **98.03%** test accuracy; AUC **0.97**.
- Classification ensemble (Bacterial vs. Viral): **91.93%** test accuracy; AUC **0.91**.
- Individual PneumoNet-v1: 96.2% (detection), 87.3% (classification).
- Individual PneumoNet-v2: 96.8% (detection).

## Research documentation

See `research/` for the paper analysis:
- `paper-notes.md` — section-by-section summary, equation glossary, evidence labels
- `method-spec.yaml` — structured method specification
- `assumptions-and-open-questions.md` — INFERRED/DEFAULT/UNKNOWN items
- `implementation-plan.md` + `TASKS.md` — task tracker with status

## License

Code: MIT (see `LICENSE` if added). The source paper is CC BY 4.0. This project is an independent reimplementation and is not affiliated with the paper's authors.

## Citation

If you use this work, please cite the original paper:

> Godbole, S., Kattukaran, A., Savla, S., Pradhan, V., Kanani, P., & Patil, D. (2024). Enhancing Paediatric Pneumonia Detection and Classification Using Customized CNNs and Transfer Learning Based Ensemble Models. *Int. Res. J. Multidisc. Technovation*, 6(6), 38–53. https://doi.org/10.54392/irjtmt2463
