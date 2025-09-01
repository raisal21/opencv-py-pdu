#!/usr/bin/env python3
"""run_accuracy_test.py (material‑detector version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Collects *coverage* predictions produced by EyeLog **MaterialDetector**—a
combination of `ForegroundExtraction` and `ContourProcessor` classes located
in `app/utils/material_detector.py`—for a folder of sample videos, then
(optionally) evaluates them against ground‑truth labels via
`accuracy_evaluator.py`.

Highlights vs previous draft
---------------------------
* **Uses the real OpenCV pipeline** (`ForegroundExtraction` + `ContourProcessor`).
* Preset‑aware: choose with `--bg-preset` & `--contour-preset` (or "all").
* No dependency on `eyelog.detector.CoverageDetector`.
* Outputs per‑frame CSV (`frame,timestamp_ms,coverage_percent,predicted`).
* Headless‑safe; CLI identical to evaluator for CI integration.

Example
~~~~~~~
```bash
python accuracy_test.py \
    --videos-dir ./samples \
    --out-dir ./preds \
    --bg-preset default \
    --contour-preset standard \
    --coverage-threshold 5.0 \
```
"""
from __future__ import annotations

import argparse, csv, datetime as _dt, json, multiprocessing as mp, sys, time
from pathlib import Path
from typing import Iterable, Tuple, List

import cv2 as cv
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Import the real detector components
# ────────────────────────────────────────────────────────────────────────────────
try:
    from .material_detector import (
        ForegroundExtraction,
        ContourProcessor,
        BG_PRESETS,
        CONTOUR_PRESETS,
        filter_coverage,
    )
    from . import material_detector as md
except ImportError as err:  # pragma: no cover – easier debug on CI
    sys.stderr.write("[FATAL] Cannot import app.utils.material_detector.\n")
    raise err
# ────────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ────────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect coverage predictions from MaterialDetector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--videos-dir", type=Path, required=True, help="Folder of videos")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for CSV + reports")
    p.add_argument("--coverage-threshold", type=float, default=5.0,
                   help="Coverage %% threshold to classify MATERIAL vs NO_MATERIAL")
    p.add_argument("--bg-preset", default="default", choices=list(BG_PRESETS) + ["all"],
                   help="Background preset name or 'all' to loop through every preset")
    p.add_argument("--contour-preset", default="standard", choices=list(CONTOUR_PRESETS) + ["all"],
                   help="Contour preset name or 'all' to test every preset")
    p.add_argument("--n-jobs", type=int, default=max(mp.cpu_count() - 1, 1))
    p.add_argument(
        "--no-coverage-filter",
        dest="apply_cov_filter",
        action="store_false",
        default=True,
        help="Disable temporal/spike filtering; use raw coverage values"
    )
    return p

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _video_pred_path(out_dir: Path, video: Path, bg: str, ct: str) -> Path:
    stem = video.stem
    return out_dir / f"{stem}__{bg}__{ct}__pred_{_timestamp()}.csv"

# ────────────────────────────────────────────────────────────────────────────────
# Core processing
# ────────────────────────────────────────────────────────────────────────────────

def _make_detector(bg_name: str, ct_name: str):
    fg = ForegroundExtraction(**BG_PRESETS[bg_name])
    ct = ContourProcessor(**CONTOUR_PRESETS[ct_name])
    return fg, ct


def _process_frame(fg: ForegroundExtraction, ct: ContourProcessor, frame) -> float:
    """Return *raw* coverage percent (0‑100) for the given frame.*"""
    fg_res = fg.process_frame(frame)
    ct_res = ct.process_mask(fg_res.binary)
    return ct_res.metrics.get("contour_coverage_percent", 0.0)


def _reset_filter_state():
    """Clear global buffers used by `filter_coverage` so each video starts fresh."""
    md.med_buf.clear()         # type: ignore[attr‑defined]
    md.ma_buf.clear()          # type: ignore[attr‑defined]
    md.prev_cov = None         # type: ignore[attr‑defined]


def process_video(video_path: Path, out_dir: Path, bg: str, ct: str, thr: float,
                  apply_filter: bool, show: bool = False) -> Tuple[Path, float]:
    csv_path = _video_pred_path(out_dir, video_path, bg, ct)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fg, contour_proc = _make_detector(bg, ct)
    _reset_filter_state()  # <‑‑ NEW: separate filter state per video

    start = time.perf_counter()
    frame_idx = 0

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_number", "timestamp_ms", "coverage_percent", "predicted_label"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_cov = _process_frame(fg, contour_proc, frame)
            if apply_filter:
                cov, _spike = filter_coverage(raw_cov) 
            else:
                cov, _spike = raw_cov, False
            pred_label = int(cov >= thr)
            ts_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
            writer.writerow([frame_idx, ts_ms, f"{cov:.2f}", pred_label])

            frame_idx += 1

    fps = frame_idx / (time.perf_counter() - start) if frame_idx else 0.0
    cap.release()
    if show:
        cv.destroyAllWindows()
    return csv_path, fps

# ────────────────────────────────────────────────────────────────────────────────
# Main entrypoint (unchanged apart from note)
# ────────────────────────────────────────────────────────────────────────────────

def main():
    args = _build_parser().parse_args()
    videos = sorted(args.videos_dir.glob("*.mp4")) + sorted(args.videos_dir.glob("*.avi"))
    if not videos:
        sys.stderr.write("[FATAL] No video files found.\n"); sys.exit(1)

    bg_choices = [args.bg_preset] if args.bg_preset != "all" else list(BG_PRESETS)
    ct_choices = [args.contour_preset] if args.contour_preset != "all" else list(CONTOUR_PRESETS)

    tasks = []
    for vid in videos:
        for bg in bg_choices:
            for ct in ct_choices:
                tasks.append((vid, args.out_dir, bg, ct, args.coverage_threshold, args.apply_cov_filter))

    print(f"[INFO] Running {len(tasks)} combinations using {args.n_jobs} workers …")

    with mp.Pool(args.n_jobs) as pool:
        results = pool.starmap(process_video, tasks)

    pred_paths, fps_list = zip(*results)
    print(f"[INFO] Mean FPS: {sum(fps_list)/len(fps_list):.2f}")


if __name__ == "__main__":
    main()
