#!/usr/bin/env python3
"""
accuracy_evaluator.py – JSON evaluator (file‑pair mode)
======================================================
Evaluates **one prediction / ground‑truth pair at a time** (no more
*directory scan*).  The script still supports both *binary* and
*quadrant* label schemes, but you now pass explicit file paths rather
than pointing to folders.

Example
-------
```
python accuracy_evaluator.py \
  --pred-json ROI_01_binary_pred.json \
  --gt-json   ROI_01_binary_gt.json   \
  --mode      binary                  \
  --out-dir   reports
```

Key changes vs directory version
--------------------------------
* `--pred-json` / `--gt-json` (required) – point directly to JSON files.
* `--mode` still accepts `binary` or `quadrant`.  If omitted, the script
  infers it from the *pred* filename (contains "binary" → binary etc.).
* Outputs:
  * `<stem>_cm.png` – colour‑blind‑friendly confusion matrix
  * `<stem>_report.json` – classification report from scikit‑learn
  * *aggregate.json* – a single‑element list capturing metrics for this
    run (keeps downstream tooling unchanged)

Dependencies
============
```
pip install numpy scikit-learn matplotlib
```
"""
from __future__ import annotations

import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────

def _read_mapping(path: Path) -> Dict[int, int]:
    """Return mapping {frame:int → label:int}."""
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): int(v) for k, v in d.items()}


def _align_frames(pred_map: Dict[int, int], gt_map: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    common = sorted(set(pred_map) & set(gt_map))
    if not common:
        raise ValueError("No overlapping frames between prediction & GT.")
    return (np.fromiter((pred_map[f] for f in common), dtype=int),
            np.fromiter((gt_map[f] for f in common), dtype=int))

# ────────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────────

def _save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: Path, mode: str):
    palette = {
        "binary": "Blues",
        "quadrant": "YlGn",
    }
    cmap = plt.get_cmap(palette.get(mode, "Blues"))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11, weight="bold")
    ax.set_ylabel("Ground‑Truth", fontsize=11, weight="bold")

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    if mode == "quadrant":
        plt.colorbar(im, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────────

def _evaluate(pred_json: Path, gt_json: Path, out_dir: Path, mode: str):
    logging.info("Evaluating %s", pred_json.name)

    y_pred_map = _read_mapping(pred_json)
    y_true_map = _read_mapping(gt_json)
    y_pred, y_true = _align_frames(y_pred_map, y_true_map)

    if mode == "infer":
        if "quadrant" in pred_json.stem.lower():
            mode_ = "quadrant"
        elif "binary" in pred_json.stem.lower():
            mode_ = "binary"
        else:
            raise ValueError("--mode not provided and cannot infer from filename")
    else:
        mode_ = mode

    if mode_ == "binary":
        labels = [0, 1]
        class_names = ["NoMaterial", "Material"]
    else:
        labels = [1, 2, 3, 4]
        class_names = [f"Q{n}" for n in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=class_names, output_dict=True)

    stem = pred_json.stem.replace("_pred", "")
    _save_confusion_matrix(cm, class_names, out_dir / f"{stem}_cm.png", mode_)
    (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {"file": pred_json.name, "mode": mode_, "accuracy": acc, "report": report}

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate EyeLog predictions vs ground‑truth (JSON)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-json", required=True, help="Prediction JSON mapping frame→label")
    p.add_argument("--gt-json",   required=True, help="Ground‑truth JSON mapping frame→label")
    p.add_argument("--mode", choices=["binary", "quadrant", "infer"], default="infer",
                   help="Label scheme; 'infer' uses filename heuristics")
    p.add_argument("--out-dir", default="reports", help="Folder for PNG/JSON outputs")
    return p


def main():
    args = _build_parser().parse_args()
    pred_json = Path(args.pred_json).expanduser()
    gt_json   = Path(args.gt_json).expanduser()
    out_dir   = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _evaluate(pred_json, gt_json, out_dir, args.mode)

    # keep downstream pipeline unchanged – wrap in list
    (out_dir / "aggregate.json").write_text(json.dumps([result], indent=2), encoding="utf-8")
    logging.info("Finished. Outputs saved in %s", out_dir.resolve())


if __name__ == "__main__":
    main()