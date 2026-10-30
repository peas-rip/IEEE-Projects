"""Replace the flawed pipeline check with the correct reconstruction.

The earlier version compared a float64 Stage 1 blend against the live float32
pipeline and then compared that pipeline figure against a *stage-1* number.
This one mirrors evaluate_full_pipeline.py exactly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PAPER_S1 = {"vgg16": 5, "inception_v3": 4, "pneumonet_v2": 3, "pneumonet_v1": 1}
S2 = ["pneumonet_v1", "mobilenet_v2", "inception_v3"]


def blend(probs_by_name, weights):
    names = [n for n in probs_by_name if n in weights]
    if not names:
        raise ValueError("no member in weights has probabilities")
    w = np.array([weights[n] for n in names], dtype=np.float32)
    w = w / w.sum()
    return sum(weight * probs_by_name[name] for weight, name in zip(w, names))


def main():
    d = np.load("results/test_predictions.npz", allow_pickle=False)
    y = d["y_true"]
    yb = d["y_true_binary"]

    s1 = {n: d[f"stage1/{n}"] for n in PAPER_S1}
    s2 = {n: d[f"stage2/{n}"] for n in S2}

    det = blend(s1, PAPER_S1)
    cls = blend(s2, {n: 1.0 for n in S2})

    det_pred = det.argmax(1)
    cls_pred = cls.argmax(1)
    pipeline_pred = np.where(det_pred == 0, 0, 1 + cls_pred)

    # Drop-Inception ablation, matching --ablate.
    drop_w = {n: w for n, w in PAPER_S1.items() if n != "inception_v3"}
    det_drop = blend(s1, drop_w)
    drop_pred = np.where(det_drop.argmax(1) == 0, 0, 1 + cls_pred)

    rows = [
        ("Stage 1 binary accuracy", (det_pred == yb).mean(), 0.8606, "stage1 ensemble"),
        ("Stage 1 missed pneumonia", int(((det_pred == 0) & (yb == 1)).sum()), 5, "count"),
        ("Pipeline 3-class accuracy", (pipeline_pred == y).mean(), 0.7853, "pipeline"),
        ("Ablation: drop Inception-v3", (drop_pred == y).mean(), 0.8301, "pipeline"),
    ]
    print(f"{'check':32s} {'local':>9s} {'shipped':>9s} {'match':>7s}   scope")
    print("-" * 76)
    ok = True
    for name, local, ref, scope in rows:
        if scope == "count":
            match = abs(int(local) - ref) <= 3
            shown = f"{int(local)}"
            ref_s = f"{ref}"
        else:
            match = abs(float(local) - ref) < 0.01
            shown = f"{local:.4f}"
            ref_s = f"{ref:.4f}"
        ok &= match
        print(f"{name:32s} {shown:>9s} {ref_s:>9s} {'OK' if match else 'DIFF':>7s}   {scope}")
    print()
    print("All checks reproduce the shipped figures." if ok else "One or more checks differ.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
