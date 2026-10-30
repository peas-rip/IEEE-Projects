"""Dump per-member probabilities over the full test set into a .npz artifact.

This is the foundation the web app's Playground page reads from. Running all
seven models once and caching the output means weight sliders, threshold
changes and ablations recompute in microseconds without touching the GPU.

Usage:
    python scripts/dump_predictions.py
    python scripts/dump_predictions.py --output results/test_predictions.npz
    python scripts/dump_predictions.py --batch-size 32
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pneumonia.checkpoints import (
    CHECKPOINTS,
    checkpoint_path,
    describe_inventory,
    root_candidates,
)
from pneumonia.data import get_transforms
from pneumonia.evaluate import predict_probs
from pneumonia.models import PneumoNetV1, PneumoNetV2
from pneumonia.transfer import TransferLearningModel


# Stage 1 members, keyed by the short name used in ensemble weighting.
STAGE1_MEMBERS = {
    "pneumonet_v1": "stage1_pneumonet_v1",
    "pneumonet_v2": "stage1_pneumonet_v2",
    "vgg16": "stage1_vgg16",
    "inception_v3": "stage1_inception",
}

# Stage 2 members. "inception_v3" is the Xception substitute (torchvision
# ships no Xception); the short name matches the ensemble scripts.
STAGE2_MEMBERS = {
    "pneumonet_v1": "stage2_pneumonet_v1",
    "mobilenet_v2": "stage2_mobilenet",
    "inception_v3": "stage2_xception",
}

TEST_ROOT = "data/real_3class/test"

#: Ground-truth order the pipeline emits: NORMAL=0, BACTERIAL=1, VIRAL=2.
#: This is NOT ImageFolder's order (which is alphabetical), which is why
#: labels must be remapped before writing the artifact.
PIPELINE_ORDER = ["NORMAL", "BACTERIAL", "VIRAL"]

#: Paper's published Stage 1 detection weights, keyed by the short names used
#: in the dump (and in the ensemble scripts).
PAPER_STAGE1_WEIGHTS = {"vgg16": 5, "inception_v3": 4, "pneumonet_v2": 3, "pneumonet_v1": 1}

#: Equal weights for Stage 2: the paper does not publish them.
STAGE2_WEIGHTS = {"pneumonet_v1": 1.0, "mobilenet_v2": 1.0, "inception_v3": 1.0}


def remap_labels(targets, class_names) -> np.ndarray:
    """Map ImageFolder targets to pipeline order (NORMAL=0, BACTERIAL=1, VIRAL=2)."""
    order = {name: idx for idx, name in enumerate(PIPELINE_ORDER)}
    return np.array([order[class_names[t]] for t in targets], dtype=np.int64)


def binary_labels(pipeline_labels) -> np.ndarray:
    """Collapse pipeline-order labels into NORMAL=0 / PNEUMONIA=1."""
    return (np.asarray(pipeline_labels) != 0).astype(np.int64)


def blend(probs_by_name: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of member probabilities, renormalised over what is present.

    Accumulates in float32 to match the live pipeline. Accumulating in float64
    shifts borderline images across the routing threshold and inflates the
    end-to-end accuracy by several points, so this is deliberate.
    """
    names = [n for n in probs_by_name if n in weights]
    if not names:
        raise ValueError("no member in weights has probabilities")
    w = np.array([weights[n] for n in names], dtype=np.float32)
    w = w / w.sum()
    return sum(weight * probs_by_name[name] for weight, name in zip(w, names))


class BinaryStage1Dataset(Dataset):
    """Collapse the 3-class test folder into NORMAL=0 / PNEUMONIA=1."""

    def __init__(self, base: ImageFolder):
        self.base = base
        self.labels = binary_labels(remap_labels(base.targets, base.classes))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img, self.labels[idx]


def load_model(arch_name: str, weights_path: str, num_classes: int = 2):
    """Build a model by architecture name and load its checkpoint."""
    if arch_name == "pneumonet_v1":
        model = PneumoNetV1(num_classes=num_classes)
    elif arch_name == "pneumonet_v2":
        model = PneumoNetV2(num_classes=num_classes)
    else:
        model = TransferLearningModel(name=arch_name, num_classes=num_classes, pretrained=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def resolve_members(mapping: dict[str, str]) -> dict[str, str]:
    """Resolve checkpoint paths for a member mapping.

    All members are required for the dump; a missing one aborts the run so
    the artifact is never silently incomplete.
    """
    resolved = {}
    for short_name, config_name in mapping.items():
        path = checkpoint_path(*CHECKPOINTS[config_name])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path} ({short_name})")
        resolved[short_name] = path
    return resolved


def dump_stage(
    mapping: dict[str, str],
    loader: DataLoader,
    device: torch.device,
    stage_label: str,
) -> dict[str, np.ndarray]:
    """Run every member of one stage over the loader and return per-member probs."""
    paths = resolve_members(mapping)
    results = {}
    for short_name, path in paths.items():
        arch = short_name  # short_name matches the architecture name
        model = load_model(arch, path).to(device)
        _, probs = predict_probs(model, loader, device=device)
        results[short_name] = probs.astype(np.float32)
        print(f"  {stage_label:7s} {short_name:15s}  {probs.shape[0]} images  {probs.shape[1]} classes")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dump per-member test-set probabilities to .npz")
    parser.add_argument("--output", default="results/test_predictions.npz",
                        help="Output .npz path (default: results/test_predictions.npz)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default: 32)")
    parser.add_argument("--data-root", default=TEST_ROOT,
                        help=f"3-class test folder (default: {TEST_ROOT})")
    args = parser.parse_args(argv)

    print(f"Checkpoint root: {root_candidates()[0]}")
    print(describe_inventory())
    print()

    if not os.path.isdir(args.data_root):
        print(f"ERROR: test data not found: {args.data_root}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test root: {args.data_root}")
    print()

    eval_tf = get_transforms(image_size=224, augment=False)[1]
    raw_ds = ImageFolder(root=args.data_root, transform=eval_tf)
    s1_ds = BinaryStage1Dataset(raw_ds)
    s1_loader = DataLoader(s1_ds, batch_size=args.batch_size, shuffle=False)
    s2_loader = DataLoader(raw_ds, batch_size=args.batch_size, shuffle=False)

    # ImageFolder sorts alphabetically: BACTERIAL=0, NORMAL=1, VIRAL=2.
    # The pipeline emits NORMAL=0, BACTERIAL=1, VIRAL=2.
    pipeline_labels = remap_labels(raw_ds.targets, raw_ds.classes)

    filenames = np.array([raw_ds.samples[i][0] for i in range(len(raw_ds))])

    print(f"Images: {len(raw_ds)}")
    print(f"Class distribution (pipeline order): NORMAL={int((pipeline_labels==0).sum())} "
          f"BACTERIAL={int((pipeline_labels==1).sum())} VIRAL={int((pipeline_labels==2).sum())}")
    print()

    print("Stage 1 members (binary: NORMAL=0, PNEUMONIA=1)")
    stage1_probs = dump_stage(STAGE1_MEMBERS, s1_loader, device, "stage1")
    print()

    print("Stage 2 members (binary: BACTERIAL=0, VIRAL=1)")
    stage2_probs = dump_stage(STAGE2_MEMBERS, s2_loader, device, "stage2")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        # Per-member probabilities, one (N, 2) float32 array per key.
        **{f"stage1/{k}": v for k, v in stage1_probs.items()},
        **{f"stage2/{k}": v for k, v in stage2_probs.items()},
        # Ground truth in pipeline order (NORMAL=0, BACTERIAL=1, VIRAL=2).
        y_true=pipeline_labels,
        # Binary labels: 0=NORMAL, 1=PNEUMONIA (what Stage 1 actually predicts).
        y_true_binary=binary_labels(pipeline_labels),
        # Source file paths, aligned with the probability rows.
        filenames=filenames,
        class_names=np.array(["NORMAL", "BACTERIAL", "VIRAL"]),
    )

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output}  ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
