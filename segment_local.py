#!/usr/bin/env python3
"""
segment_local.py

Run MobileSAM segmentation locally (no server). Produces:
  - <basename>_overlay.png
  - <basename>_segments.json
  - <basename>_segments.csv

By default it applies your "large-like-0" rule:
  * Compute stats ONLY inside the UNION of all bboxes whose area >= 75% of box-0 area
  * Labels are drawn only for segments that have >=1 pixel inside that ROI

Use --full to compute stats on the full masks (no bbox restriction).
"""

import os
import sys
import json
import csv
import argparse
import pathlib
from PIL import Image

# Ensure relative imports work even if script is run from elsewhere
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Import your existing logic
from main import (                 # noqa: E402
    analyze_segments,
    analyze_segments_in_bbox0,
)

# Reuse helpers if you want (they're defined in main.py)
from main import _ensure_dir as ensure_dir  # noqa: E402
from main import _save_stats_csv as save_stats_csv  # noqa: E402
from main import _save_stats_json as save_stats_json  # noqa: E402


def process_one_image(input_path: str, outdir: str, inside_bbox0: bool = True):
    """
    Process a single image and save overlay + stats to outdir.
    """
    p = pathlib.Path(input_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    ensure_dir(outdir)

    # Build output paths
    stem = p.stem
    base = pathlib.Path(outdir) / stem
    overlay_path = f"{base}_overlay.png"
    csv_path = f"{base}_segments.csv"
    json_path = f"{base}_segments.json"

    # Load image
    img = Image.open(p).convert("RGB")

    # Analyze
    if inside_bbox0:
        overlay, stats = analyze_segments_in_bbox0(img)
    else:
        overlay, stats = analyze_segments(img)

    # Save results
    overlay.save(overlay_path)
    save_stats_csv(stats, csv_path)
    save_stats_json(stats, json_path)

    # Console summary
    print("\n=== Segmentation complete ===")
    print(f"Input:   {p}")
    print(f"Overlay: {overlay_path}")
    print(f"JSON:    {json_path}")
    print(f"CSV:     {csv_path}")
    print(f"Segments in report: {len(stats)}")


def main():
    ap = argparse.ArgumentParser(
        description="Run MobileSAM segmentation locally (no HTTP), producing overlay + stats."
    )
    ap.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input image file (e.g., resources/train/02_JPG...jpg)"
    )
    ap.add_argument(
        "-o", "--outdir",
        default="output",
        help="Directory to write results (default: ./output)"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Use full-mask statistics (no bbox restriction). By default we use the 'large-like-0' rule."
    )

    args = ap.parse_args()
    inside_bbox0 = not args.full

    try:
        process_one_image(args.input, args.outdir, inside_bbox0=inside_bbox0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

