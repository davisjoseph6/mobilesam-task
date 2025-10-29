#!/usr/bin/env python3
"""
batch_local.py

Offline batch processor that mirrors /batch/process-resources, but saves to:
  - output/train
  - output/valid

It scans resources/train and resources/valid for acceptable images, runs the same
MobileSAM pipeline used by the API, and writes for each input image:
  - <basename>_overlay.png
  - <basename>_segments.json
  - <basename>_segments.csv

By default, it applies the "large-like-0" rule (stats only inside the union of
all bboxes whose area >= 75% of the area of segment-0's bbox).
Use --full to compute stats over full masks (no bbox restriction).

Examples
--------
# Default: process both train and valid with the 'large-like-0' rule
python batch_local.py

# Only process 'train'
python batch_local.py --subset train

# Process both subsets but compute full-mask stats (no bbox restriction)
python batch_local.py --full
"""

import os
import sys
import csv
import json
import pathlib
import argparse
from typing import Dict, List

from PIL import Image

# Ensure we run from project root so that "main" imports model/weights correctly.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Reuse your existing analysis functions (which load the model on import)
from main import (  # noqa: E402
    analyze_segments,
    analyze_segments_in_bbox0,
)

# ---------- Constants (local batch output targets) ----------
RESOURCES_DIR = pathlib.Path("resources")
OUTPUT_DIR = pathlib.Path("output")
SUBSETS = ("train", "valid")
ACCEPT_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_image_file(p: pathlib.Path) -> bool:
    return p.is_file() and p.suffix.lower() in ACCEPT_EXT


def _save_stats_csv(stats: List[Dict], csv_path: pathlib.Path) -> None:
    _ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pixels", "color_class", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for s in stats:
            bx, by, bw, bh = (s.get("bbox_xywh") or [None, None, None, None])
            writer.writerow([s["id"], s["pixels"], s["color_class"], bx, by, bw, bh])


def _save_stats_json(stats: List[Dict], json_path: pathlib.Path) -> None:
    _ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _process_subset(subset: str, inside_bbox0: bool) -> List[Dict]:
    """
    Process all acceptable images under resources/<subset> and write outputs to output/<subset>.
    Returns a list of item dicts like the API manifest does.
    """
    src_dir = RESOURCES_DIR / subset
    out_dir = OUTPUT_DIR / subset
    _ensure_dir(out_dir)

    items: List[Dict] = []
    if not src_dir.exists():
        print(f"[INFO] resources/{subset} does not exist; skipping.")
        return items

    # Walk recursively to preserve folder structure
    for p in src_dir.rglob("*"):
        if not _is_image_file(p):
            continue

        rel = p.relative_to(src_dir)
        stem = p.stem
        base = out_dir / rel.parent / stem

        overlay_path = base.with_name(f"{stem}_overlay").with_suffix(".png")
        csv_path = base.with_name(f"{stem}_segments").with_suffix(".csv")
        json_path = base.with_name(f"{stem}_segments").with_suffix(".json")

        img = Image.open(p).convert("RGB")
        if inside_bbox0:
            overlay, stats = analyze_segments_in_bbox0(img)
        else:
            overlay, stats = analyze_segments(img)

        _ensure_dir(overlay_path.parent)
        overlay.save(overlay_path)
        _save_stats_csv(stats, csv_path)
        _save_stats_json(stats, json_path)

        items.append({
            "input": str(p),
            "overlay": str(overlay_path),
            "csv": str(csv_path),
            "json": str(json_path),
        })

        print(f"[OK] {p} -> {overlay_path.name}, {json_path.name}, {csv_path.name}")

    return items


def main():
    ap = argparse.ArgumentParser(description="Local batch processor for MobileSAM (no API).")
    ap.add_argument(
        "--subset",
        choices=SUBSETS,
        nargs="+",
        default=list(SUBSETS),
        help="Which subsets to process (default: train valid)"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Compute stats over full masks (no bbox restriction). Default uses the 'large-like-0' rule."
    )
    args = ap.parse_args()

    inside_bbox0 = not args.full

    manifest: Dict[str, List[Dict]] = {}
    for subset in args.subset:
        print(f"\n=== Processing subset: {subset} (inside_bbox0={inside_bbox0}) ===")
        manifest[subset] = _process_subset(subset, inside_bbox0=inside_bbox0)

    # Print manifest JSON (mirrors the API’s return structure)
    print("\n=== Manifest ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

