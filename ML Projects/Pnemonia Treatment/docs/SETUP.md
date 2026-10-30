# Setup

This guide gets you from a clean machine to the numbers in
[`RESULTS.md`](RESULTS.md). There are two tracks:

* **Track A — evaluate the trained checkpoints.** No GPU required (a GPU makes
  it faster). This is what the shipped results were produced with.
* **Track B — retrain all seven models.** Needs a GPU; expect a long Colab
  session. Only do this if you want to change the method.

Everything else (figures, docs, exercises) works in Track A.

---

## 1. Requirements

| Item | Version | Notes |
|---|---|---|
| Python | 3.9+ | 3.11 or 3.13 tested |
| pip | any | |
| Disk | ~1.5 GB | ~700 MB checkpoints + ~550 MB images + cache |
| GPU | optional | CPU evaluation of 624 images takes minutes, not seconds |
| OS | Windows / Linux / macOS | Windows: set `--num-workers 0` if workers hang |

Dependencies are pinned loosely in `requirements.txt`: `torch`, `torchvision`,
`scikit-learn`, `Pillow`, `numpy`, `matplotlib`, `pyyaml`, `tqdm`.

---

## 2. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Verify:

```bash
python -c "import torch, torchvision, sklearn; print(torch.__version__, torchvision.__version__)"
pytest -q
```

Expected: **44 passed**. If tests fail, stop here — the environment is not
right and nothing downstream will be trustworthy.

> The package is importable as `pneumonia` and exposes a console script:
> `pneumonia --help`, `pneumonia train`, `pneumonia evaluate`, `pneumonia infer`.

---

## 3. Directory layout the scripts expect

```
Pnemonia Treatment/
├── configs/                     # one YAML per model
├── data/
│   ├── real/                    # raw Kermany download (source of truth)
│   │   └── chest_xray/
│   │       ├── train/           # 5,216 images
│   │       ├── test/            #   624 images
│   │       └── val/             #    16 images (never used)
│   ├── real_3class/             # what Stage 1 trains on
│   │   ├── train/               # NORMAL 1,341 · BACTERIAL 2,530 · VIRAL 1,345
│   │   └── test/                # NORMAL   234 · BACTERIAL   242 · VIRAL   148
│   └── real_stage2/             # what Stage 2 trains on
│       ├── train/               # BACTERIAL 2,530 · VIRAL 1,345
│       └── test/                # BACTERIAL   242 · VIRAL   148
├── results/
│   ├── stage1_pneumonet_v1/final_model.pt
│   ├── stage1_pneumonet_v2/final_model.pt
│   ├── stage1_vgg16/final_model.pt
│   ├── stage1_inception/final_model.pt
│   ├── stage2_pneumonet_v1/final_model.pt
│   ├── stage2_mobilenet/final_model.pt
│   ├── stage2_xception/final_model.pt      # the Inception-v3 substitute
│   └── reproduction_run.json
└── scripts/
```

The seven checkpoint paths are the only hard-coded expectations. They are
resolved by `src/pneumonia/checkpoints.py`, which tries:

1. `$PNEUMONIA_CHECKPOINT_DIR/<name>/final_model.pt`
2. `results/<name>/final_model.pt`
3. legacy directory names (e.g. `stage2_checkpoints` for `stage2_pneumonet_v1`)

So on Colab you can leave the files on Drive and just point at them:

```bash
export PNEUMONIA_CHECKPOINT_DIR=/content/drive/MyDrive/PneumoniaProject/checkpoints
```

On Windows PowerShell:

```powershell
$env:PNEUMONIA_CHECKPOINT_DIR = "D:\checkpoints"
```

**There is no `val/` folder on disk.** The validation split is created at
runtime as 20% of each `train/` folder (`train_ratio: 0.8`, `seed: 42`). That is
why Stage 1 reports 4,172 train / 1,044 validation.

---

## 4. Get the data

The dataset is **not** included — it is the public Kermany paediatric chest
X-ray corpus (Kaggle: `paultimothymooney/chest-xray-pneumonia`, derived from
Guangzhou Women and Children's Medical Center). Review its licence before
redistributing anything derived from it.

Download and unpack so you have a `chest_xray/` folder containing `train/`,
`test/`, `val/`, then:

```bash
python scripts/prepare_dataset.py --source path/to/chest_xray --dst data --clean
```

This writes `data/real/chest_xray/`, `data/real_3class/` and
`data/real_stage2/`. Confirm the counts:

```bash
# Linux / macOS
find data/real_3class/test -name "*.jpeg" | wc -l      # 624
find data/real_stage2/test -name "*.jpeg" | wc -l      # 390
```

```powershell
# Windows
(Get-ChildItem data\real_3class\test -Recurse -File).Count   # 624
(Get-ChildItem data\real_stage2\test -Recurse -File).Count   # 390
```

Optional but recommended: pre-resize to a 224 px cache so every epoch does not
re-decode full-resolution JPEGs.

```bash
python scripts/prepare_cache.py --src data/real_3class --dst data/cache_3class_224 --size 224
python scripts/prepare_cache.py --src data/real_stage2 --dst data/cache_stage2_224 --size 224
```

The evaluation scripts read `data/real_3class/test` and `data/real_stage2/test`
directly, so the cache only speeds up training.

---

## 5. Get the checkpoints

Place the seven `.pt` files in the layout shown in Section 3, or point
`PNEUMONIA_CHECKPOINT_DIR` at wherever they already are.

Each file is a plain `state_dict` (or a training checkpoint containing
`model_state_dict`; both are accepted). A quick sanity check:

```bash
python -c "
import torch
s = torch.load('results/stage1_vgg16/final_model.pt', map_location='cpu')
print(len(s), 'tensors; first key:', next(iter(s)))
"
```

If you do **not** have the checkpoints, use Track B (Section 8). The evaluation
scripts will tell you which path they looked for:

```
MISSING weights: results/stage1_vgg16/final_model.pt (vgg16)
```

---

## 6. Run the evaluations

Run from the repository root.

```bash
# Stage 1: Normal vs Pneumonia
python scripts/evaluate_stage1_ensemble.py --json results/stage1_ensemble.json

# Stage 2: Bacterial vs Viral
python scripts/evaluate_stage2_ensemble.py --json results/stage2_ensemble.json

# Full pipeline + ablations
python scripts/evaluate_full_pipeline.py --ablate --json results/pipeline_ablation.json
```

`--allow-missing` runs with whatever members are present and renormalises their
weights — useful for experimenting, never for reporting.

`--json` writes a timestamped file with the same schema as
`results/reproduction_run.json`. `--ablate` adds the zero-retraining
comparisons (Section 7 of `RESULTS.md`).

### Known-good numbers

Check your output against these. Anything else means the checkpoints or the data
layout differ from the shipped run.

| Run | Expected |
|---|---:|
| Stage 1 ensemble accuracy | 0.8606 |
| Stage 1 ensemble AUC | 0.9634 |
| Stage 2 ensemble accuracy | 0.8949 |
| Stage 2 ensemble AUC | 0.9390 |
| Pipeline accuracy | 0.7853 |
| Pipeline AUC | 0.9503 |
| Ablation: drop Inception-v3 | 0.8301 |
| Ablation: Stage 2 weights 2/3/5 | 0.7981 |
| Ablation: both | 0.8429 |

Small differences (≤ 0.0005) are just rounding. Differences at the first decimal
are not.

---

## 7. Single-image inference

```bash
# Stage 1 detection with a PneumoNet checkpoint (default architecture)
pneumonia infer --image path/to/xray.jpeg --checkpoint results/stage1_pneumonet_v1/final_model.pt

# Any architecture: pick it explicitly or point at its config
pneumonia infer --image path/to/xray.jpeg \
  --checkpoint results/stage1_vgg16/final_model.pt \
  --model vgg16 --labels Normal,Pneumonia

pneumonia infer --image path/to/xray.jpeg \
  --checkpoint results/stage2_mobilenet/final_model.pt \
  --config configs/stage2_mobilenet.yaml --labels BACTERIAL,VIRAL
```

`--config` supplies the model name, input size and class count; `--model`,
`--image-size` and `--labels` override individual pieces. The image is converted
to grayscale then replicated to 3 channels, matching training.

The CLI scores **one model**, not the ensemble. For the full pipeline, use
`scripts/evaluate_full_pipeline.py` (or write a single-image wrapper as an
exercise — see `docs/EXERCISES.md`).

---

## 8. Track B — retrain from scratch (optional)

Order matters: Stage 1 checkpoints feed Stage 1 evaluation, Stage 2 checkpoints
feed Stage 2 evaluation, and the pipeline needs all seven.

```bash
# Stage 1 (Normal vs Pneumonia)
python -m pneumonia train --config configs/stage1_pneumonet_v1.yaml --data data/real_3class/train --checkpoint-dir results/stage1_pneumonet_v1
python -m pneumonia train --config configs/stage1_pneumonet_v2.yaml --data data/real_3class/train --checkpoint-dir results/stage1_pneumonet_v2
python -m pneumonia train --config configs/stage1_vgg16.yaml        --data data/real_3class/train --checkpoint-dir results/stage1_vgg16
python -m pneumonia train --config configs/stage1_inception.yaml    --data data/real_3class/train --checkpoint-dir results/stage1_inception

# Stage 2 (Bacterial vs Viral)
python -m pneumonia train --config configs/stage2_pneumonet_v1.yaml --data data/real_stage2/train --checkpoint-dir results/stage2_pneumonet_v1
python -m pneumonia train --config configs/stage2_mobilenet.yaml    --data data/real_stage2/train --checkpoint-dir results/stage2_mobilenet
python -m pneumonia train --config configs/stage2_xception.yaml     --data data/real_stage2/train --checkpoint-dir results/stage2_xception
```

Notes:

* `--data` must point at an ImageFolder root (one sub-folder per class). The
  80/20 train/validation split happens inside the loader.
* Transfer-learning backbones download ImageNet weights on first use. On a
  sandboxed machine, pre-download `torch` hub weights or expect the first run to
  fail.
* `--no-amp` disables mixed precision; use it if you hit AMP numerical issues.
* Training writes `checkpoint_epoch*.pt` plus `final_model.pt` into
  `--checkpoint-dir`. Evaluation uses `final_model.pt`.
* `configs/stage2_xception.yaml` trains an **Inception-v3**, because torchvision
  has no Xception. The file and directory keep the paper's name deliberately so
  the substitution stays visible.

Expect the retrained numbers to differ from `RESULTS.md` unless you match the
seed, the GPU, and the library versions.

---

## 9. Regenerate the figures

```bash
python scripts/make_figures.py
```

Reads `results/reproduction_run.json`, writes five PNGs to `docs/figures/`.
It does not need PyTorch or the checkpoints — only `matplotlib` and `numpy`, so
it works on any machine.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MISSING weights: .../final_model.pt` | Checkpoint not where the script looks | Set `PNEUMONIA_CHECKPOINT_DIR`, or move files into the layout in Section 3 |
| `ValueError: Target is multiclass but average='binary'` | Old Stage 2 breakdown bug (a `-1` sentinel from shifting predictions without shifting labels) | Update to this version — fixed in `4edab95` |
| `IndentationError` / import errors after `pip install -e .` | Stale `build/` or `.egg-info` | Delete them and reinstall |
| `DataLoader` hangs or crashes on Windows | `num_workers > 0` in a script run | Append `--num-workers 0` where supported, or set `data.num_workers: 0` in the config |
| CUDA out of memory | Batch too large for the GPU | Lower `data.batch_size` in the config |
| `FutureWarning: default weight initialization of inception_v3` | torchvision notice, printed even with pretrained weights | Harmless; ignore |
| `UnpicklingError` / `Weights only load failed` | torch ≥ 2.6 defaults to `weights_only=True` | Pass a plain `state_dict`, or `torch.load(..., weights_only=False)` for trusted checkpoint dicts |
| Low accuracy on all models | Evaluating the wrong test folder (e.g. `data/real` instead of `data/real_3class`) | Check the paths in Section 3; expect 624 and 390 test images |
| `FileNotFoundError: data/real_3class/test` | Dataset not prepared | Section 4 |

### Class order — get this right or every number is wrong

`torchvision.datasets.ImageFolder` sorts class folders **alphabetically**:

```
BACTERIAL = 0, NORMAL = 1, VIRAL = 2
```

The pipeline emits its own order:

```
NORMAL = 0, BACTERIAL = 1, VIRAL = 2
```

`evaluate_full_pipeline.py` remaps ground truth and prints:

```
Remapped ground-truth labels from ImageFolder order ['BACTERIAL', 'NORMAL', 'VIRAL']
to pipeline order ['NORMAL', 'BACTERIAL', 'VIRAL']
```

If you write your own evaluation code and skip that remap, you will report a
confusion matrix that looks like a broken model. Both Stage 1 and Stage 2
scripts avoid the issue by constructing binary/2-class loaders directly.

---

## 11. Working without a GPU

Everything in Track A runs on CPU; it is just slower. Reasonable settings:

```yaml
data:
  batch_size: 8
  num_workers: 0
```

The scripts hard-code the batch size in a few places (`DatasetConfig(..., batch_size=16)`),
so if CPU memory is tight, run one script at a time. Figure generation
(`scripts/make_figures.py`) is CPU-only by design and takes a second.
