#!/usr/bin/env python3
"""
metrics_exported.py — exporter for quadrant metrics (fixed Q1–Q4)
================================================================
Reads prediction and ground-truth annotations from JSON mapping
`{frame:int -> label}` (label may be 1..4 or "Q1".."Q4"), aligns the
common frames, computes confusion-matrix based metrics (precision, recall,
F1), and exports compact CSV/JSON artifacts for downstream visualization.

This script **does not render any plots** and **does not compute IoU**.
It only emits the data needed for a separate visualizer, plus fixed
quadrant bounding boxes for every frame (to allow the visualizer to
compute IoU in a flexible way).

Outputs
-------
OUT_DIR/
  ├─ per_frame.csv                # frame,pred_q,gt_q
  ├─ confusion_matrix.csv         # rows=GT(Q1..Q4), cols=PRED(Q1..Q4)
  ├─ classification_metrics.json  # per_class + micro/macro/weighted + accuracy
  ├─ boxes.csv                    # per-frame fixed boxes for pred & gt
  └─ meta.json                    # bookkeeping info, ROI, mapping, counts

Usage
-----
python metrics_exported.py \
  --pred-json run1_pred.json \
  --gt-json   run1_gt.json \
  --out-dir   out/run1 \
  --roi 0,0,1,1         # normalized ROI (default)

JSON schema (input)
-------------------
{ "123": 2, "124": 1, ... }      # frame → quadrant (1..4 or "Q1".."Q4")
"""
from __future__ import annotations

import argparse, json, logging, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

CLASS_ORDER = ["Q1", "Q2", "Q3", "Q4"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_ORDER, start=1)}  # "Q1"->1,..
ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}

# ────────────────────────────────────────────────────────────────────────────────
# ROI utilities
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class ROI:
    x: float
    y: float
    w: float
    h: float
    normalized: bool = True  # default normalized (0..1)

    @staticmethod
    def parse(s: str | None) -> "ROI":
        if not s:
            return ROI(0.0, 0.0, 1.0, 1.0, normalized=True)
        parts = [float(v.strip()) for v in s.split(",")]
        if len(parts) != 4:
            raise ValueError("--roi must be 'x,y,w,h'")
        norm = all(0.0 <= v <= 1.0 for v in parts)
        return ROI(parts[0], parts[1], parts[2], parts[3], normalized=norm)

def quad_box(q: int, roi: ROI) -> Tuple[float, float, float, float]:
    """Return (x1,y1,x2,y2) of fixed box for quadrant q within ROI (same units as ROI)."""
    if q not in (1, 2, 3, 4):
        raise ValueError(f"Invalid quadrant: {q}")
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    midx, midy = x + w / 2.0, y + h / 2.0
    if q == 1:   # top-left
        return (x, y, midx, midy)
    if q == 2:   # top-right
        return (midx, y, x + w, midy)
    if q == 3:   # bottom-left
        return (x, midy, midx, y + h)
    # q == 4     # bottom-right
    return (midx, midy, x + w, y + h)

# ────────────────────────────────────────────────────────────────────────────────
# I/O & label normalization
# ────────────────────────────────────────────────────────────────────────────────

def read_mapping(path: Path) -> Dict[int, int]:
    """Read JSON {frame -> label} and normalize labels to integers 1..4."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping: Dict[int, int] = {}
    for k, v in raw.items():
        frame = int(k)
        # normalize value
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean.startswith("Q"):
                v_clean = v_clean[1:]
            label = int(v_clean)
        else:
            label = int(v)
        if label not in (1, 2, 3, 4):
            raise ValueError(f"Label must be 1..4 or Q1..Q4, got {v!r}")
        mapping[frame] = label
    return mapping

def align_frames(pred_map: Dict[int, int], gt_map: Dict[int, int]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Return (sorted_common_frames, y_pred, y_true)."""
    common = sorted(set(pred_map) & set(gt_map))
    if not common:
        raise ValueError("No overlapping frames between prediction & GT.")
    y_pred = np.fromiter((pred_map[f] for f in common), dtype=int)
    y_true = np.fromiter((gt_map[f] for f in common), dtype=int)
    return common, y_pred, y_true

# ────────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────────

def confusion_matrix_4(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute 4x4 confusion matrix (rows=GT Q1..Q4, cols=Pred Q1..Q4)."""
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t) - 1, int(p) - 1] += 1
    return cm

def metrics_from_cm(cm: np.ndarray) -> dict:
    """Return per-class precision/recall/f1/support and aggregated averages."""
    assert cm.shape == (4, 4), "confusion matrix must be 4x4"
    per_class = {}
    totals = cm.sum()
    diag_sum = int(np.trace(cm))
    # vectors
    support = cm.sum(axis=1)             # per GT class (row sums)
    pred_sums = cm.sum(axis=0)           # per Pred class (col sums)

    precisions, recalls, f1s, supports = [], [], [], []

    for idx, name in enumerate(CLASS_ORDER):
        tp = cm[idx, idx]
        fp = pred_sums[idx] - tp
        fn = support[idx] - tp
        p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        s = int(support[idx])
        per_class[name] = {"precision": p, "recall": r, "f1": f1, "support": s}
        precisions.append(p); recalls.append(r); f1s.append(f1); supports.append(s)

    # Averages
    total_support = sum(supports) if len(supports) else 0
    def wavg(values, weights):
        if total_support == 0:
            return 0.0
        return float(np.average(values, weights=weights))

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall":    float(np.mean(recalls)) if recalls else 0.0,
        "f1":        float(np.mean(f1s)) if f1s else 0.0,
    }
    weighted = {
        "precision": wavg(precisions, supports),
        "recall":    wavg(recalls, supports),
        "f1":        wavg(f1s, supports),
    }
    micro_acc = (diag_sum / totals) if totals > 0 else 0.0  # equals micro P/R/F1 in balanced multiclass
    out = {
        "per_class": per_class,
        "averages": {
            "micro": {"precision": micro_acc, "recall": micro_acc, "f1": micro_acc},
            "macro": macro,
            "weighted": weighted,
        },
        "accuracy": micro_acc,
        "support_total": int(totals),
    }
    return out

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def export_all(pred_json: Path, gt_json: Path, out_dir: Path, roi: ROI, labels: List[str] | None = None) -> None:
    labels = labels or CLASS_ORDER
    if labels and len(labels) == 4:
        display_labels = dict(zip(CLASS_ORDER, labels))
    else:
        display_labels = {k: k for k in CLASS_ORDER}

    pred_map = read_mapping(pred_json)
    gt_map = read_mapping(gt_json)

    frames_common, y_pred, y_true = align_frames(pred_map, gt_map)
    cm = confusion_matrix_4(y_true, y_pred)
    metrics = metrics_from_cm(cm)

    # per_frame.csv
    per_frame_path = out_dir / "per_frame.csv"
    per_frame_path.parent.mkdir(parents=True, exist_ok=True)
    with per_frame_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "pred_q", "gt_q"])
        for fr, p, t in zip(frames_common, y_pred, y_true):
            w.writerow([fr, int(p), int(t)])

    # confusion_matrix.csv (rows=GT Q1..Q4, cols=Pred Q1..Q4)
    cm_path = out_dir / "confusion_matrix.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["GT\\PRED"] + CLASS_ORDER)
        for i, name in enumerate(CLASS_ORDER):
            row = [name] + [int(v) for v in cm[i].tolist()]
            w.writerow(row)

    # classification_metrics.json
    (out_dir / "classification_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # boxes.csv (fixed boxes for pred & gt)
    boxes_path = out_dir / "boxes.csv"
    with boxes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "pred_q", "gt_q",
            "pred_x1", "pred_y1", "pred_x2", "pred_y2",
            "gt_x1",   "gt_y1",  "gt_x2",   "gt_y2",
            "coord_space"
        ])
        coord_space = "normalized" if roi.normalized else "pixel"
        for fr, p, t in zip(frames_common, y_pred, y_true):
            pb = quad_box(int(p), roi)
            gb = quad_box(int(t), roi)
            w.writerow([fr, int(p), int(t), *[f"{v:.6f}" for v in pb], *[f'{v:.6f}' for v in gb], coord_space])

    # meta.json
    meta = {
        "pred_json": str(pred_json),
        "gt_json": str(gt_json),
        "num_frames_pred": len(pred_map),
        "num_frames_gt": len(gt_map),
        "num_frames_common": len(frames_common),
        "dropped_pred_only": len(set(pred_map) - set(frames_common)),
        "dropped_gt_only": len(set(gt_map) - set(frames_common)),
        "class_order": CLASS_ORDER,
        "roi": {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h, "normalized": roi.normalized},
        "notes": "Confusion matrix rows=GT(Q1..Q4), cols=Pred(Q1..Q4). Labels normalized to 1..4.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logging.info("Exported: %s", out_dir.resolve())

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export metrics artifacts (fixed quadrant mode) for downstream visualization."
    )
    p.add_argument("--pred-json", required=True, help="Prediction JSON mapping {frame -> quadrant}")
    p.add_argument("--gt-json",   required=True, help="Ground-truth JSON mapping {frame -> quadrant}")
    p.add_argument("--out-dir",   required=True, help="Output directory for CSV/JSON artifacts")
    p.add_argument("--roi", default="0,0,1,1", help="ROI as x,y,w,h (normalized if all ≤ 1, else pixel units)")
    p.add_argument("--labels", default="Q1,Q2,Q3,Q4", help="(Optional) display names (still fixed order).")
    return p

def main():
    args = build_parser().parse_args()
    pred_json = Path(args.pred_json).expanduser()
    gt_json = Path(args.gt_json).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    roi = ROI.parse(args.roi)
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else CLASS_ORDER
    export_all(pred_json, gt_json, out_dir, roi, labels)

if __name__ == "__main__":
    main()
