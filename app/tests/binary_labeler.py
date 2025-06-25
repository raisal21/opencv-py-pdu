# ================================
# File: binary_labeler.py
# ================================
#!/usr/bin/env python3
"""
binary_labeler.py – JSON‑driven binary label reviewer
======================================================
Review binary (0/1) labels that already exist in a JSON mapping of
*exact* frame numbers.  The script **only** seeks those frames, shows
an OpenCV window, and asks the operator to confirm or override them.

Added CLI flags
---------------
--max-samples N  : review at most *N* frames (0 = all)

Example
-------
python binary_labeler.py \
       --video sample.mp4 \
       --json  ROI_01_binary_pred.json \
       --out   revised_labels.json \
       --max-samples 750
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2 as cv

# ----------------------------------------------------------------------------- utilities

def load_mapping(json_path: Path) -> Dict[int, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.items()}


def save_mapping(mapping: Dict[int, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in sorted(mapping.items())}, f, indent=2)
    print(f"\n✅ Revised labels saved to {out_path}")


# ----------------------------------------------------------------------------- main routine

def run_reviewer(args: argparse.Namespace) -> None:
    allowed_vals: List[int] = [int(v) for v in args.values]

    mapping = load_mapping(Path(args.json))
    frame_list = sorted(mapping)

    if args.max_samples and args.max_samples > 0:
        frame_list = frame_list[: args.max_samples]
        print(f"Limiting review to {len(frame_list)} / {len(mapping)} frames.")

    cap = cv.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    revised = dict(mapping)
    for idx, frame_no in enumerate(frame_list, start=1):
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            print(f"⚠️  Failed to read frame {frame_no}. Skipping.")
            continue

        cv.imshow("binary_labeler", frame)
        cv.waitKey(1)

        default_val = mapping[frame_no]
        prompt = f"[{idx}/{len(frame_list)}] Apakah frame {frame_no} bernilai [{default_val}]? "
        while True:
            try:
                answer = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer == "":
                new_val = default_val
                break
            if answer.isdigit() and int(answer) in allowed_vals:
                new_val = int(answer)
                break
            print(f"Input tidak valid. Masukkan salah satu dari {allowed_vals} atau tekan Enter.")

        if new_val != default_val:
            revised[frame_no] = new_val

    cap.release()
    cv.destroyAllWindows()

    out_path = Path(args.out) if args.out else Path(args.json).with_name(Path(args.json).stem + "_revised.json")
    save_mapping(revised, out_path)


# ----------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="binary_labeler.py",
        description="Review binary labels on frames listed in a JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--json", required=True, help="Path to JSON mapping frame→label")
    p.add_argument("--out", default=None, help="Output JSON path (default: <json>_revised.json)")
    p.add_argument("--values", nargs="+", default=[0, 1], help="Allowed label values")
    p.add_argument("--max-samples", type=int, default=0, help="Max frames to review; 0 = all")
    return p


def main():
    run_reviewer(build_parser().parse_args())


if __name__ == "__main__":
    main()