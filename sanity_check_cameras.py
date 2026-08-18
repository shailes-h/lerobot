#!/usr/bin/env python
"""Standalone helper: grab one frame from each RealSense camera and save it to disk,
so you can eyeball the framing/alignment without starting a full record/inference run.

Usage (defaults match the 3-camera setup in README_REAL_ROBOT.md):
    python capture_alignment_frame.py

Override serials / output dir if needed:
    python capture_alignment_frame.py \
        --right 260322275072 --left 260322271881 --top 262522074294 \
        --out_dir ./alignment_frames
"""

import argparse
from pathlib import Path

import cv2

from lerobot.cameras import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right", default="260322275072", help="Right camera serial number")
    parser.add_argument("--left", default="260322271881", help="Left camera serial number")
    parser.add_argument("--top", default="262522074294", help="Top camera serial number")
    parser.add_argument("--out_dir", default="./alignment_frames", help="Where to save the captured frames")
    args = parser.parse_args()

    cameras = {
        "right": RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=args.right, width=640, height=480, fps=30, color_mode=ColorMode.BGR
            )
        ),
        "left": RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=args.left, width=640, height=480, fps=30, color_mode=ColorMode.BGR
            )
        ),
        "top": RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=args.top, width=640, height=360, fps=30, color_mode=ColorMode.BGR
            )
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cam in cameras.items():
        print(f"Connecting {name} ({cam})...")
        cam.connect()
        try:
            frame = cam.read()  # BGR, ready for cv2.imwrite
            out_path = out_dir / f"{name}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved {out_path} ({frame.shape[1]}x{frame.shape[0]})")
        finally:
            cam.disconnect()

    print(f"\nDone. Frames saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
