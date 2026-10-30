# Running the Web App

The app is a full-stack web application: a FastAPI backend serving a React
single-page frontend. It runs fully offline once the package is unzipped.

## Prerequisites

- Python 3.9+ with pip
- The seven checkpoint files in `checkpoints/` (included in the package)
- `results/test_predictions.npz` (included in the package — 36 KB)
- ~30 sample images in `samples/` (included)

No Node.js, no GPU, no internet connection required to run the app.

## Quick start

### Windows

```bat
run_app.bat
```

### Linux / macOS

```bash
chmod +x run_app.sh
./run_app.sh
```

### Manual

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
pip install -e ".[serve]"

python app.main
```

The browser opens at `http://localhost:8000`. The API docs are at
`http://localhost:8000/docs`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `APP_HOST` | `127.0.0.1` | Bind address |
| `APP_PORT` | `8000` | Port |
| `PNEUMONIA_CHECKPOINT_DIR` | `checkpoints` | Override checkpoint location (Colab/Drive) |

## What the app does

### Predict page

Drag-and-drop an image or pick from the 30 curated samples. The two-stage
pipeline classifies it as Normal, Bacterial, or Viral, with per-member
probability bars and re-weighting presets (paper weights, equal weights, drop
Inception-v3).

### Playground page

Live weight sliders for all 7 members, a Stage 1 threshold slider, and
ablation toggles. The accuracy and confusion matrix update instantly from
cached test-set predictions — no model runs. This is the demo feature: a
student can show that dropping Inception-v3 raises accuracy from 0.7853 to
0.8301 in real time.

### Results page

The shipped figures and tables from `results/reproduction_run.json`, including
the gap analysis and deviation ledger. Honest framing: 78.53% end-to-end, not
98.03%.

### About page

Limitations, source paper, deviations from the paper, and the educational-use
licence.

## How it works

```
Browser (React SPA)
   |
   |  HTTP
   v
FastAPI (app/main.py)
   |
   |-- /api/predict    -> PipelineService.predict()  (7 forward passes, ~0.4s)
   |-- /api/reblend    -> blend cached probs          (~0.0000s)
   |-- /api/testset/evaluate -> score from .npz       (~0.001s)
   |-- /api/samples    -> list curated images
   |-- /api/figures/*  -> serve docs/figures/*.png
   |-- /api/results    -> serve reproduction_run.json
   |-- /api/health     -> checkpoint status, device, versions
   |
   v
pneumonia.service (PipelineService)
   |
   |-- loads 7 checkpoints once (3.0s, first predict only)
   |-- runs Stage 1 (4 models) + Stage 2 (3 models)
   |-- blends with weighted average of softmax probabilities
```

The `.npz` artifact (`results/test_predictions.npz`, 36 KB) holds per-member
probabilities for all 624 test images. The Playground reads this instead of
running inference, so slider moves are instant.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Weights not found" on health page | `checkpoints/` missing or empty | Check that the ZIP was fully extracted; see `docs/SETUP.md` §3 |
| App opens but page is blank | `frontend/dist/` missing | Run `cd frontend && npm install && npm run build` |
| Predict takes > 5 seconds | CPU-only machine | Install CUDA-enabled PyTorch; see `docs/SETUP.md` §1 |
| `ModuleNotFoundError: app` | Running from wrong directory | `cd` to the project root first |
| Port 8000 in use | Another process | `APP_PORT=8001 python app.main` |

## Building the frontend (for developers)

The shipped `frontend/dist/` is pre-built. To modify the UI:

```bash
cd frontend
npm install
npm run dev     # dev server with hot reload on :5173, proxies /api to :8000
npm run build   # rebuilds frontend/dist/
```

Node.js 18+ is required for development. Buyers never need it.
