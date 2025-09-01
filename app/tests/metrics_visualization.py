#!/usr/bin/env python3
"""
metrics_visualization.py — render charts from exporter artifacts
================================================================
Consumes the CSV/JSON files emitted by `metrics_exported.py` and renders:
  1) Bar chart of Precision / Recall / F1 per class (Q1..Q4)
  2) Confusion matrix heatmap (rows=GT, cols=Pred)
  3) IoU analytics:
       - Compute IoU per-frame from boxes.csv
       - Render mean IoU per quadrant (grouped by GT) + overall mean annotation

Outputs (PNG files) are written into `--out-dir` (default: metrics dir).
"""
from __future__ import annotations

import argparse, json, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt

CLASS_ORDER = ["Q1", "Q2", "Q3", "Q4"]

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def ensure_out_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def load_classification_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def load_confusion_matrix(path: Path) -> np.ndarray:
    # Expect CSV with header ["GT\\PRED", Q1, Q2, Q3, Q4] and 4 rows
    data = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            data.append([int(v) for v in row[1:5]])
    cm = np.array(data, dtype=int)
    assert cm.shape == (4, 4), f"confusion_matrix.csv malformed: got {cm.shape}"
    return cm

def load_boxes(path: Path):
    frames, pred_q, gt_q = [], [], []
    pred_boxes, gt_boxes = [], []
    coord_space = None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            pred_q.append(int(row["pred_q"]))
            gt_q.append(int(row["gt_q"]))
            pb = (float(row["pred_x1"]), float(row["pred_y1"]), float(row["pred_x2"]), float(row["pred_y2"]))
            gb = (float(row["gt_x1"]), float(row["gt_y1"]), float(row["gt_x2"]), float(row["gt_y2"]))
            pred_boxes.append(pb); gt_boxes.append(gb)
            coord_space = row.get("coord_space", coord_space)
    return np.array(frames), np.array(pred_q), np.array(gt_q), np.array(pred_boxes), np.array(gt_boxes), coord_space

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0

# ────────────────────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────────────────────

def plot_prf_bar(metrics: dict, out_png: Path) -> None:
    per = metrics["per_class"]
    # ordered arrays
    P = [per[c]["precision"] for c in CLASS_ORDER]
    R = [per[c]["recall"]    for c in CLASS_ORDER]
    F = [per[c]["f1"]        for c in CLASS_ORDER]

    x = np.arange(len(CLASS_ORDER))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar(x - width, P, width, label="Precision")
    ax.bar(x,         R, width, label="Recall")
    ax.bar(x + width, F, width, label="F1-score")

    ax.set_xticks(x, CLASS_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per Class")
    ax.legend(loc="upper right")

    # annotate bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    fig.tight_layout()
    ensure_out_dir(out_png.parent)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix(cm: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    im = ax.imshow(cm, aspect="equal")
    ax.set_xticks(np.arange(4), CLASS_ORDER)
    ax.set_yticks(np.arange(4), CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (rows=GT, cols=Pred)")

    # annotate
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_out_dir(out_png.parent)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_iou_stats(frames, gt_q, pred_boxes, gt_boxes, out_png: Path) -> None:
    # IoU per frame
    ious = np.array([iou_xyxy(a, b) for a, b in zip(pred_boxes, gt_boxes)], dtype=float)
    overall_mean = float(np.mean(ious)) if len(ious) else 0.0

    # Mean IoU per quadrant (by GT)
    means = []
    for q in [1, 2, 3, 4]:
        mask = (gt_q == q)
        means.append(float(np.mean(ious[mask])) if np.any(mask) else 0.0)

    x = np.arange(len(CLASS_ORDER))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.bar(x, means)
    ax.set_xticks(x, CLASS_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean IoU")
    ax.set_title("Mean IoU per Quadrant (grouped by GT)")

    # annotate bars and overall
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.text(0.98, 0.02, f"Overall mean IoU: {overall_mean:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    fig.tight_layout()
    ensure_out_dir(out_png.parent)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render metrics visualizations from exporter artifacts.")
    p.add_argument("--metrics-dir", required=True, help="Directory produced by metrics_exported.py")
    p.add_argument("--out-dir", default=None, help="Output directory for PNGs (default: metrics-dir)")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    return p

def main():
    args = build_parser().parse_args()
    mdir = Path(args.metrics_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else mdir
    ensure_out_dir(out_dir)

    # 1) PRF bar
    metrics = load_classification_metrics(mdir / "classification_metrics.json")
    plot_prf_bar(metrics, out_dir / "prf_bar.png")

    # 2) Confusion matrix
    cm = load_confusion_matrix(mdir / "confusion_matrix.csv")
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    # 3) IoU stats
    frames, pred_q, gt_q, pred_boxes, gt_boxes, coord = load_boxes(mdir / "boxes.csv")
    plot_iou_stats(frames, gt_q, pred_boxes, gt_boxes, out_dir / "iou_stats.png")

if __name__ == "__main__":
    main()
